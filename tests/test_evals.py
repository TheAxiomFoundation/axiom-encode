"""Tests for model comparison eval helpers."""

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
import yaml

from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.evals import (
    EvalArtifactMetrics,
    EvalContextFile,
    EvalPromptResponse,
    EvalReadinessGates,
    EvalResult,
    EvalSuiteCase,
    EvalSuiteManifest,
    EvalWorkspace,
    GroundingMetric,
    _build_empty_artifact_retry_prompt,
    _build_eval_prompt,
    _canonical_target_ref_prefix,
    _clean_generated_file_content,
    _codex_prompt_timeouts,
    _command_looks_out_of_bounds,
    _context_file_executable_surfaces,
    _eval_result_from_payload,
    _evaluate_generated_artifact_with_repairs,
    _format_subparagraph_coverage_checklist,
    _hydrate_eval_root,
    _is_single_amount_table_slice,
    _materialize_eval_artifact,
    _normalize_nonannual_test_period_value,
    _normalize_test_case_value,
    _normalize_test_periods_to_effective_dates,
    _post_openai_eval_request,
    _resolve_eval_output_path,
    _resolve_eval_reference_source_id,
    _rulespec_validation_target,
    _run_codex_prompt_eval,
    _select_cross_section_context_files,
    _source_identifier_to_relative_rulespec_path,
    _target_source_scope_for_heuristics,
    _wait_for_codex_process,
    evaluate_artifact,
    find_admin_agency_aggregate_entity_issues,
    load_eval_suite_manifest,
    parse_runner_spec,
    prepare_eval_workspace,
    resolve_corpus_source_unit,
    run_eval_suite,
    run_model_eval,
    run_source_eval,
    select_context_files,
    summarize_readiness,
)
from axiom_encode.harness.validator_pipeline import (
    ValidationResult,
    ValidatorPipeline,
    find_test_input_assignment_issues,
)


@pytest.fixture(autouse=True)
def _mock_generalist_reviewer():
    """Keep eval tests deterministic unless they explicitly inspect reviewer behavior."""
    with patch.object(
        ValidatorPipeline,
        "_run_reviewer",
        return_value=ValidationResult(
            "generalist-reviewer", True, score=8.0, issues=[]
        ),
    ):
        yield


def _write_test_corpus_provision(
    tmp_path: Path,
    citation_path: str = "us/statute/7/2017",
    body: str = "authoritative source text",
) -> Path:
    corpus_path = tmp_path / "axiom-corpus"
    parts = citation_path.split("/")
    provisions_dir = (
        corpus_path / "data" / "corpus" / "provisions" / parts[0] / parts[1]
    )
    provisions_dir.mkdir(parents=True, exist_ok=True)
    (provisions_dir / "test.jsonl").write_text(
        json.dumps({"citation_path": citation_path, "body": body}) + "\n",
        encoding="utf-8",
    )
    return corpus_path


class TestParseRunnerSpec:
    def test_parses_named_runner(self):
        runner = parse_runner_spec("gpt=codex:gpt-5.4")
        assert runner.name == "gpt"
        assert runner.backend == "codex"
        assert runner.model == "gpt-5.4"

    def test_parses_default_name(self):
        runner = parse_runner_spec("claude:opus")
        assert runner.name == "claude-opus"
        assert runner.backend == "claude"
        assert runner.model == "opus"

    def test_parses_openai_runner(self):
        runner = parse_runner_spec("openai:gpt-5.4")
        assert runner.name == "openai-gpt-5.4"
        assert runner.backend == "openai"
        assert runner.model == "gpt-5.4"


def test_source_identifier_maps_corpus_regulation_to_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us-ny/regulation/18-nycrr/387/12/f/3/v/c"
    ) == Path("regulations/18-nycrr/387/12/f/3/v/c.yaml")


def test_source_identifier_maps_colon_prefixed_regulation_to_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us-co:regulations/10-ccr-2506-1/4.804.1"
    ) == Path("regulations/10-ccr-2506-1/4.804.1.yaml")


def test_source_identifier_maps_state_manual_to_policies_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us-az/manual/des/faa5/na-child-support-expense/block-2"
    ) == Path("policies/des/faa5/na-child-support-expense/block-2.yaml")


def test_admin_agency_aggregate_rejects_household_executable_rule():
    source_text = (
        "FNS shall estimate each State agency's active case, payment, and negative "
        "case error rate. y2′ = y2 + b2(X2−x2), where X2 is the average value of "
        "allotments underissued to participating households in the State agency "
        "full quality control sample."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/275/23
rules:
  - name: average_allotments_underissued_active_error_rate
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    source: 7 CFR 275.23(b)(2)(i)(B)
    versions:
      - effective_from: '0001-01-01'
        formula: y2 + b2 * (x2_full_sample - x2_rereview)
"""

    issues = find_admin_agency_aggregate_entity_issues(content, source_text)

    assert issues == [
        "Unsupported administrative aggregate entity: "
        "`average_allotments_underissued_active_error_rate` is declared on "
        "`Household`, but the authoritative source defines a State agency/FNS "
        "aggregate performance, sampling, liability, waiver, or bonus measure. "
        "Use a source-stated administrative entity such as `StateAgency` "
        "instead of a household/person/tax/payment entity, or defer only if the "
        "administrative surface still cannot be represented faithfully."
    ]


def test_admin_agency_aggregate_rejects_bonus_payment_spending_restriction():
    source_text = (
        "Bonus payments shall not be used for household benefits, including "
        "incentive payments."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/275/24
rules:
  - name: bonus_payment_may_be_used_for_household_benefits
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 7 CFR 275.24(a)(8)(i)
    versions:
      - effective_from: '0001-01-01'
        formula: not payment_is_bonus_payment or not payment_use_is_household_benefit
"""

    issues = find_admin_agency_aggregate_entity_issues(content, source_text)

    assert issues == [
        "Unsupported administrative aggregate entity: "
        "`bonus_payment_may_be_used_for_household_benefits` is declared on "
        "`Payment`, but the authoritative source defines a State agency/FNS "
        "aggregate performance, sampling, liability, waiver, or bonus measure. "
        "Use a source-stated administrative entity such as `StateAgency` "
        "instead of a household/person/tax/payment entity, or defer only if the "
        "administrative surface still cannot be represented faithfully."
    ]


def test_admin_agency_aggregate_allows_state_agency_entity():
    source_text = (
        "The amount of the liability shall be equal to the product of the value "
        "of all allotments issued by the State agency, the difference between "
        "the State agency's payment error rate and 6 percent, and 10 percent."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/275/23
rules:
  - name: state_agency_payment_error_rate_liability
    kind: derived
    entity: StateAgency
    dtype: Money
    period: Year
    source: 7 CFR 275.23(d)(2)
    versions:
      - effective_from: '2003-10-01'
        formula: all_allotments_issued_by_state_agency * (state_agency_payment_error_rate - 0.06) * 0.10
"""

    assert find_admin_agency_aggregate_entity_issues(content, source_text) == []


def test_admin_agency_aggregate_allows_household_level_source():
    source_text = (
        "A household is eligible for SNAP if it meets the household income "
        "standard and resource test."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/273/9
rules:
  - name: household_snap_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: 7 CFR 273.9
    versions:
      - effective_from: '0001-01-01'
        formula: household_income_eligible and household_resource_eligible
"""

    assert find_admin_agency_aggregate_entity_issues(content, source_text) == []


def test_admin_agency_aggregate_allows_long_income_exclusion_list():
    source_text = (
        "P.L. No. 100-175, Section 166, Older Americans Act. Funds received by "
        "persons fifty-five (55) years of age and older under the Senior "
        "Community Service Employment Program under Title V of the Older "
        "Americans Act are excluded from income. State agencies and eight "
        "organizations receive funding under Title V. "
        + ("Separate income exclusion text. " * 40)
        + "P.L. No. 101-508 amended Section 402(i) of the Social Security Act. "
        "At-risk block grant child care payments are excluded from being "
        "counted as income for SNAP purposes."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-co/regulation/10-ccr-2506-1/4.405.2
rules:
  - name: payment_excluded_as_income
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Month
    source: 10 CCR 2506-1, 4.405.2
    versions:
      - effective_from: '0001-01-01'
        formula: senior_community_service_payment or at_risk_child_care_payment
"""

    assert find_admin_agency_aggregate_entity_issues(content, source_text) == []


def test_source_identifier_maps_federal_regulation_to_cfr_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us/regulation/7/273/10"
    ) == Path("regulations/7-cfr/273/10.yaml")


def test_source_identifier_maps_federal_form_to_allowed_policy_repo_path():
    assert _source_identifier_to_relative_rulespec_path(
        "us/form/cms/medicaid-chip-bhp-eligibility-levels"
    ) == Path("policies/cms/medicaid-chip-bhp-eligibility-levels.yaml")


def test_resolve_corpus_source_unit_accepts_form_citation_path(tmp_path):
    citation = "us/form/cms/medicaid-chip-bhp-eligibility-levels"
    corpus_path = _write_test_corpus_provision(
        tmp_path,
        citation_path=citation,
        body="CMS Medicaid, CHIP, and BHP eligibility levels table",
    )

    source_unit = resolve_corpus_source_unit(citation, corpus_path)

    assert source_unit.citation_path == citation
    assert source_unit.source == "local"
    assert source_unit.body == "CMS Medicaid, CHIP, and BHP eligibility levels table"


def test_resolve_corpus_source_unit_slices_before_bracketed_sibling(tmp_path):
    corpus_path = _write_test_corpus_provision(
        tmp_path,
        citation_path="us/statute/26/3306",
        body=(
            "(k) Agricultural labor For purposes of this chapter, the term "
            "agricultural labor has the meaning assigned by section 3121(g).\n\n"
            "[(l) Repealed. Sept. 1, 1954.]\n\n"
            "(m) American vessel and aircraft For purposes of this chapter, "
            "the term American vessel means a documented vessel."
        ),
    )

    source_unit = resolve_corpus_source_unit("26 USC 3306(k)", corpus_path)

    assert source_unit.citation_path == "us/statute/26/3306"
    assert source_unit.body.startswith("(k) Agricultural labor")
    assert "[(l) Repealed" not in source_unit.body
    assert "(m) American vessel" not in source_unit.body


def test_resolve_corpus_source_unit_slices_bracketed_repealed_subsection(tmp_path):
    corpus_path = _write_test_corpus_provision(
        tmp_path,
        citation_path="us/statute/26/3306",
        body=(
            "(k) Agricultural labor For purposes of this chapter, the term "
            "agricultural labor has the meaning assigned by section 3121(g).\n\n"
            "[(l) Repealed. Sept. 1, 1954.]\n\n"
            "(m) American vessel and aircraft For purposes of this chapter, "
            "the term American vessel means a documented vessel."
        ),
    )

    source_unit = resolve_corpus_source_unit("26 USC 3306(l)", corpus_path)

    assert source_unit.citation_path == "us/statute/26/3306"
    assert source_unit.body == "[(l) Repealed. Sept. 1, 1954.]"


def test_resolve_corpus_source_unit_uses_form_child_blocks(tmp_path):
    citation = "us/form/cms/medicaid-chip-bhp-eligibility-levels"
    corpus_path = tmp_path / "axiom-corpus"
    provisions_dir = corpus_path / "data" / "corpus" / "provisions" / "us" / "form"
    provisions_dir.mkdir(parents=True)
    (provisions_dir / "test.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "citation_path": citation,
                        "body": None,
                        "heading": "Medicaid, CHIP, and BHP Eligibility Levels",
                        "level": 1,
                        "ordinal": 1,
                    }
                ),
                json.dumps(
                    {
                        "citation_path": f"{citation}/block-1",
                        "body": "Colorado 142% 142% 142% 260% 195% 260% 68% 133%",
                        "heading": "State Medicaid, CHIP and BHP Income Eligibility Standards",
                        "level": 2,
                        "ordinal": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    source_unit = resolve_corpus_source_unit(citation, corpus_path)

    assert source_unit.citation_path == citation
    assert source_unit.source == "local"
    assert (
        "State Medicaid, CHIP and BHP Income Eligibility Standards" in source_unit.body
    )
    assert "Colorado 142% 142% 142% 260% 195% 260% 68% 133%" in source_unit.body


def test_canonical_target_ref_prefix_handles_canonical_source_id():
    assert (
        _canonical_target_ref_prefix(
            "us:regulations/7-cfr/273/9/d/6/iii",
            Path("regulations/7-cfr/273/9/d/6/iii.yaml"),
        )
        == "us:regulations/7-cfr/273/9/d/6/iii"
    )


def test_canonical_target_ref_prefix_uses_policy_repo_for_repo_relative_source_id(
    tmp_path,
):
    repo = tmp_path / "rulespec-us-ny"
    repo.mkdir()

    assert (
        _canonical_target_ref_prefix(
            "regulations/18-nycrr/387/14/a/1",
            Path("regulations/18-nycrr/387/14/a/1.yaml"),
            policy_repo_path=repo,
        )
        == "us-ny:regulations/18-nycrr/387/14/a/1"
    )


def test_canonical_target_ref_prefix_omits_repo_relative_source_without_repo():
    assert (
        _canonical_target_ref_prefix(
            "regulations/18-nycrr/387/14/a/1",
            Path("regulations/18-nycrr/387/14/a/1.yaml"),
        )
        is None
    )


def test_subparagraph_coverage_checklist_requires_exact_corpus_source_keys():
    checklist = _format_subparagraph_coverage_checklist(
        "(a) First category is eligible.\n(b) Second category is ineligible.",
        "us-ny/regulation/18-nycrr/387/14/a/5",
    )

    assert "copy the relevant string exactly" in checklist
    assert "human-readable source like `18 NYCRR 387.14(a)(5)(i)(a)`" in checklist
    assert "us-ny/regulation/18-nycrr/387/14/a/5(a)" in checklist


@pytest.mark.parametrize(
    "citation,expected",
    [
        # Issue #71: dot-separated CDSS-style citations must keep subsection
        # identity in the output path. Before the fix, every sibling collapsed
        # onto the section-level path because pathlib's with_suffix() treated
        # the dotted leaf as a file extension.
        ("us-ca/regulation/mpp/63-503", "regulations/mpp/63-503.yaml"),
        ("us-ca/regulation/mpp/63-503.1", "regulations/mpp/63-503/1.yaml"),
        ("us-ca/regulation/mpp/63-503.131", "regulations/mpp/63-503/131.yaml"),
        ("us-ca/regulation/mpp/63-503.132", "regulations/mpp/63-503/132.yaml"),
        ("us-ca/regulation/mpp/63-300.234", "regulations/mpp/63-300/234.yaml"),
        # Deeper dotted nesting must also split correctly.
        (
            "us-ca/regulation/mpp/63-503.131.a",
            "regulations/mpp/63-503/131/a.yaml",
        ),
        # Colorado CCR section numbers use dotted file stems in rulespec-us-co;
        # splitting the leaf would create a parallel wrong tree like 4/207/2.yaml.
        (
            "us-co/regulation/10-ccr-2506-1/4.207.2",
            "regulations/10-ccr-2506-1/4.207.2.yaml",
        ),
        (
            "us-co/regulation/10-ccr-2506-1/4.403.11/b/c/3",
            "regulations/10-ccr-2506-1/4.403.11/b/c/3.yaml",
        ),
        # Colorado statutes also use dotted legal labels for sections and
        # subsections; those labels are file stems, not nested MPP-style parts.
        (
            "us-co/statute/39/39-22-104.5",
            "statutes/39/39-22-104.5.yaml",
        ),
        (
            "us-co/statute/39/39-22-104/1.5",
            "statutes/39/39-22-104/1.5.yaml",
        ),
        # Slash-separated citations (USC, NYCRR, CFR) are unaffected — these
        # are regression cases for the dot-stripping fix.
        (
            "us-ny/regulation/18-nycrr/387/14/a/1",
            "regulations/18-nycrr/387/14/a/1.yaml",
        ),
        ("us/statute/26/1/a/1", "statutes/26/1/a/1.yaml"),
        ("us/regulation/7/273/8", "regulations/7-cfr/273/8.yaml"),
        (
            "uk/statute/legislation.gov.uk/ukpga/1992/4/section/8",
            "statutes/ukpga/1992/4/8.yaml",
        ),
        (
            "uk/regulation/legislation.gov.uk/uksi/2013/376/regulation/36",
            "regulations/uksi/2013/376/36.yaml",
        ),
        (
            "uk/regulation/legislation.gov.uk/uksi/2013/376/schedule/5/paragraph/2",
            "regulations/uksi/2013/376/schedule/5/paragraph/2.yaml",
        ),
    ],
)
def test_source_identifier_handles_dotted_leaf_segments(citation, expected):
    assert str(_source_identifier_to_relative_rulespec_path(citation)) == expected


def test_resolve_eval_output_path_uses_path_like_citation_directly():
    """Sanity: a citation that already looks like a corpus path is used as-is."""
    from axiom_encode.harness.evals import _resolve_eval_output_path

    assert _resolve_eval_output_path("us/statute/7/2014/e/2/B") == Path(
        "statutes/7/2014/e/2/B.yaml"
    )


def test_resolve_eval_output_path_uses_repo_relative_source_root_directly():
    """Repo-relative logical targets are valid --source-id output paths."""
    from axiom_encode.harness.evals import _resolve_eval_output_path

    assert _resolve_eval_output_path(
        "policies/otda/snap/fy-2026-benefit-calculation",
        requested_source="us/guidance/usda/fns/snap-fy2026-cola/page-1",
    ) == Path("policies/otda/snap/fy-2026-benefit-calculation.yaml")


def test_resolve_eval_output_path_uses_colon_prefixed_rulespec_source_id(tmp_path):
    repo = tmp_path / "rulespec-us-co"
    repo.mkdir()

    source_id = "us-co:regulations/10-ccr-2506-1/4.804.1"
    relative_output = _resolve_eval_output_path(source_id)

    assert relative_output == Path("regulations/10-ccr-2506-1/4.804.1.yaml")
    assert (
        _canonical_target_ref_prefix(
            source_id,
            relative_output,
            policy_repo_path=repo,
        )
        == "us-co:regulations/10-ccr-2506-1/4.804.1"
    )


def test_resolve_eval_output_path_uses_requested_source_when_citation_is_free_text():
    """Free-text source_id falls through to requested_source for path derivation.

    Surfaced live by us_snap_earned_income_deduction_refresh.yaml on
    2026-05-27 and us_snap_asset_test_current_effective_refresh.yaml.
    The benchmarks supply a human-readable source_id like
    "SNAP earned income deduction under 7 USC 2014(e)(2)(B)" and a
    path-like corpus_citation_path. Before the fix, the resolver only
    looked at the citation (free-text), couldn't parse it as a corpus
    path or USC citation, and landed the artifact at
    `source/<slug>.yaml` — outside any rulespec source-root directory,
    so downstream compile validators couldn't find it.
    """
    assert _resolve_eval_output_path(
        "SNAP earned income deduction under 7 USC 2014(e)(2)(B)",
        requested_source="us/statute/7/2014/e/2/B",
    ) == Path("statutes/7/2014/e/2/B.yaml")


def test_eval_reference_source_id_uses_requested_source_with_free_text_citation(
    tmp_path,
):
    repo = tmp_path / "rulespec-us-ny"
    repo.mkdir()

    target_ref_source = _resolve_eval_reference_source_id(
        "New York SNAP utility allowance",
        requested_source="regulations/18-nycrr/387/14/a/1",
    )
    relative_output = _resolve_eval_output_path(
        "New York SNAP utility allowance",
        requested_source="regulations/18-nycrr/387/14/a/1",
    )

    assert target_ref_source == "regulations/18-nycrr/387/14/a/1"
    assert relative_output == Path("regulations/18-nycrr/387/14/a/1.yaml")
    assert (
        _canonical_target_ref_prefix(
            target_ref_source,
            relative_output,
            policy_repo_path=repo,
        )
        == "us-ny:regulations/18-nycrr/387/14/a/1"
    )


def test_resolve_eval_output_path_ignores_requested_source_when_also_free_text():
    """If both inputs are free-text, fall back to the existing USC parser
    (which may itself error out — that's a separate bug, not this fix's job).
    """
    # citation is path-like USC; requested_source is also path-like but
    # different — citation wins because it's path-like.
    assert _resolve_eval_output_path(
        "us/statute/26/63",
        requested_source="us/statute/7/2014",
    ) == Path("statutes/26/63.yaml")


class TestCorpusSourceResolution:
    def test_resolves_state_manual_corpus_path_without_statute_rewrite(self, tmp_path):
        corpus_path = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-az/manual/des/faa5/na-child-support-expense/block-2",
            body="manual child support text",
        )

        source = resolve_corpus_source_unit(
            "us-az/manual/des/faa5/na-child-support-expense/block-2",
            corpus_path,
        )

        assert source.citation_path == (
            "us-az/manual/des/faa5/na-child-support-expense/block-2"
        )
        assert source.body == "manual child support text"
        assert source.source == "local"

    def test_resolves_state_statute_child_path_to_sliced_section_provision(
        self, tmp_path
    ):
        corpus_path = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(1.7) (a) A prior rate applies.\n"
                "(b) A second prior rate applies.\n"
                "(c) Except as otherwise provided, a tax of four and "
                "forty one-hundredths percent is imposed.\n"
                "(2) Federal taxable income shall be modified before the rate."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/1.7/c",
            corpus_path,
        )

        assert source.citation_path == "us-co/statute/39/39-22-104"
        assert source.body == (
            "(c) Except as otherwise provided, a tax of four and "
            "forty one-hundredths percent is imposed."
        )
        assert "(2) Federal taxable income" not in source.body

    def test_resolves_state_statute_child_path_stops_at_dotted_alpha_sibling(
        self, tmp_path
    ):
        corpus_path = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(4) There shall be subtracted from federal taxable income:\n"
                "(a) United States obligation interest included in federal "
                "taxable income.\n"
                "(a.5) Repealed.\n"
                "(b) Basis adjustment subtraction.\n"
                "(3) There shall be added to federal taxable income:\n"
                "(p) Itemized deduction addback.\n"
                "(p.5) Healthy school meals deduction addback.\n"
                "(p.5) Alternate healthy school meals deduction addback.\n"
                "(p.7) Additional healthy school meals deduction addback.\n"
                "(q) Food and beverage expense addback."
            ),
        )

        us_interest = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/4/a",
            corpus_path,
        )
        itemized_addback = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/3/p",
            corpus_path,
        )
        healthy_school_meals_addback = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/3/p/5",
            corpus_path,
        )

        assert us_interest.body == (
            "(a) United States obligation interest included in federal taxable income."
        )
        assert "(a.5) Repealed" not in us_interest.body
        assert itemized_addback.body == "(p) Itemized deduction addback."
        assert "(p.5) Healthy school meals" not in itemized_addback.body
        assert healthy_school_meals_addback.body.startswith(
            "(p.5) Healthy school meals deduction addback."
        )
        assert (
            "(p.5) Alternate healthy school meals" in healthy_school_meals_addback.body
        )
        assert (
            "(p.7) Additional healthy school meals"
            not in healthy_school_meals_addback.body
        )
        assert "(q) Food and beverage" not in healthy_school_meals_addback.body

    def test_resolves_nested_child_before_dotted_sibling_fallback(self, tmp_path):
        corpus_path = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(4) There shall be subtracted from federal taxable income:\n"
                "(a) United States obligation interest.\n"
                "(1) Nested qualifying amount.\n"
                "(a.1) Dotted sibling subtraction.\n"
                "(b) Basis adjustment subtraction."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/4/a/1",
            corpus_path,
        )

        assert source.body == "(1) Nested qualifying amount."
        assert "(a.1) Dotted sibling" not in source.body

    def test_resolves_alpha_child_path_stops_at_later_omitted_sibling(self, tmp_path):
        corpus_path = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(1) Tax is imposed as follows:\n"
                "(a) First rate period.\n"
                "(c) Third rate period after omitted subsection."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/1/a",
            corpus_path,
        )

        assert source.body == "(a) First rate period."
        assert "(c) Third rate period" not in source.body

    def test_resolves_numeric_child_path_stops_at_later_omitted_sibling(self, tmp_path):
        corpus_path = _write_test_corpus_provision(
            tmp_path,
            citation_path="us-co/statute/39/39-22-104",
            body=(
                "(1) First addition rule.\n"
                "(3) Third addition rule after omitted subsection."
            ),
        )

        source = resolve_corpus_source_unit(
            "us-co/statute/39/39-22-104/1",
            corpus_path,
        )

        assert source.body == "(1) First addition rule."
        assert "(3) Third addition rule" not in source.body

    def test_resolves_usc_child_citation_to_sliced_section_provision(self, tmp_path):
        corpus_path = tmp_path / "axiom-corpus"
        provisions_dir = (
            corpus_path / "data" / "corpus" / "provisions" / "us" / "statute"
        )
        provisions_dir.mkdir(parents=True)
        (provisions_dir / "2026-01-01.jsonl").write_text(
            json.dumps(
                {
                    "citation_path": "us/statute/26/3101",
                    "body": (
                        "(a) Old-age, survivors, and disability insurance "
                        "states 6.2 percent.\n\n"
                        "(b) Hospital insurance states 1.45 percent."
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        source = resolve_corpus_source_unit("26 USC 3101(a)", corpus_path)

        assert source.citation_path == "us/statute/26/3101"
        assert source.body == (
            "(a) Old-age, survivors, and disability insurance states 6.2 percent."
        )
        assert source.source == "local"

    def test_resolves_nested_usc_child_citation_to_sliced_section_provision(
        self, tmp_path
    ):
        corpus_path = tmp_path / "axiom-corpus"
        provisions_dir = (
            corpus_path / "data" / "corpus" / "provisions" / "us" / "statute"
        )
        provisions_dir.mkdir(parents=True)
        (provisions_dir / "2026-01-01.jsonl").write_text(
            json.dumps(
                {
                    "citation_path": "us/statute/7/2015",
                    "body": (
                        "(a) General eligibility.\n\n"
                        "(d) Work requirements (1) Paragraph one. "
                        "(2) Exemptions (A) First exemption. "
                        "(B) Second exemption. "
                        "(C) Student exemption states 20 hours. "
                        "(D) Next exemption. "
                        "(3) Other work rule.\n\n"
                        "(e) Students."
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        source = resolve_corpus_source_unit("7 USC 2015(d)(2)(C)", corpus_path)

        assert source.citation_path == "us/statute/7/2015"
        assert source.body == "(C) Student exemption states 20 hours."

    def test_resolves_nested_cfr_child_path_to_sliced_section_provision(self, tmp_path):
        corpus_path = tmp_path / "axiom-corpus"
        provisions_dir = (
            corpus_path / "data" / "corpus" / "provisions" / "us" / "regulation"
        )
        provisions_dir.mkdir(parents=True)
        (provisions_dir / "2026-01-01.jsonl").write_text(
            json.dumps(
                {
                    "citation_path": "us/regulation/7/273/9",
                    "body": (
                        "(a) Income standards.\n\n"
                        "(d) Deductions. "
                        "(5) Child support deduction.\n\n"
                        "(i) Not a top-level sibling. "
                        "(6) Shelter costs--"
                        "(i) Homeless shelter deduction. "
                        "(ii) Excess shelter deduction. "
                        "(iii) Standard utility allowances. "
                        "(A) Utility standard. "
                        "(1) Heating. (2) Cooling. (3) Other utilities.\n\n"
                        "(e) Benefit calculation."
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        source = resolve_corpus_source_unit(
            "us/regulation/7/273/9/d/6",
            corpus_path,
        )

        assert source.citation_path == "us/regulation/7/273/9"
        assert source.body.startswith("(6) Shelter costs")
        assert "(3) Other utilities" in source.body
        assert "(5) Child support deduction" not in source.body
        assert "(e) Benefit calculation" not in source.body

    def test_nested_slicing_ignores_parenthetical_cross_reference_list(self, tmp_path):
        corpus_path = tmp_path / "axiom-corpus"
        provisions_dir = (
            corpus_path / "data" / "corpus" / "provisions" / "us" / "statute"
        )
        provisions_dir.mkdir(parents=True)
        (provisions_dir / "2026-01-01.jsonl").write_text(
            json.dumps(
                {
                    "citation_path": "us/statute/26/63",
                    "body": (
                        "(c) Standard deduction "
                        "(4) Adjustments for inflation Each dollar amount "
                        "contained in paragraph (2)(B), (2)(C), or (5) or "
                        "subsection (f) shall be increased. "
                        "(5) Limitation on basic standard deduction in the case "
                        "of certain dependents In the case of an individual with "
                        "respect to whom a deduction under section 151 is "
                        "allowable to another taxpayer, the basic standard "
                        "deduction shall not exceed the greater of— (A) $500, "
                        "or (B) the sum of $250 and earned income. "
                        "(6) Certain individuals not eligible."
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        source = resolve_corpus_source_unit("26 USC 63(c)(5)", corpus_path)

        assert source.citation_path == "us/statute/26/63"
        assert source.body.startswith("(5) Limitation on basic standard deduction")
        assert "paragraph (2)(B)" not in source.body

    def test_nested_slicing_ignores_plural_parenthetical_cross_reference_list(
        self, tmp_path
    ):
        corpus_path = tmp_path / "axiom-corpus"
        provisions_dir = (
            corpus_path / "data" / "corpus" / "provisions" / "us" / "statute"
        )
        provisions_dir.mkdir(parents=True)
        (provisions_dir / "2026-01-01.jsonl").write_text(
            json.dumps(
                {
                    "citation_path": "us/statute/26/63",
                    "body": (
                        "(c) Standard deduction "
                        "(4) Adjustments for inflation Each dollar amount "
                        "contained in paragraphs (2)(B), (2)(C), or (5) or "
                        "subsection (f) shall be increased. "
                        "(5) Limitation on basic standard deduction in the case "
                        "of certain dependents In the case of an individual with "
                        "respect to whom a deduction under section 151 is "
                        "allowable to another taxpayer, the basic standard "
                        "deduction shall not exceed the greater of— (A) $500, "
                        "or (B) the sum of $250 and earned income. "
                        "(6) Certain individuals not eligible."
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        source = resolve_corpus_source_unit("26 USC 63(c)(5)", corpus_path)

        assert source.citation_path == "us/statute/26/63"
        assert source.body.startswith("(5) Limitation on basic standard deduction")
        assert "paragraphs (2)(B)" not in source.body

    def test_build_prompt_requires_resolved_corpus_locator(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="26 USC 3101(a)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Section text states 6.2 percent.",
            axiom_rules_path=tmp_path / "rulespec-us",
            mode="cold",
            source_metadata_payload={
                "corpus_citation_path": "us/statute/26/3101",
            },
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 3101(a)",
            "cold",
            workspace,
            [],
            target_file_name="a.yaml",
            include_tests=True,
            runner_backend="codex",
        )

        assert "read from `corpus.provisions` at `us/statute/26/3101`" in prompt
        assert (
            "module.source_verification.corpus_citation_path: us/statute/26/3101"
            in prompt
        )
        assert "Do not emit `source_url`" in prompt

    def test_workspace_writes_corpus_source_metadata_payload(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="26 USC 3101(a)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Section text states 6.2 percent.",
            axiom_rules_path=tmp_path / "rulespec-us",
            mode="cold",
            source_metadata_payload={
                "source_name": "Federal Insurance Contributions Act",
                "corpus_citation_path": "us/statute/26/3101",
            },
            extra_context_paths=[],
        )

        assert workspace.source_metadata == {
            "source_name": "Federal Insurance Contributions Act",
            "corpus_citation_path": "us/statute/26/3101",
        }

    def test_generation_result_fails_when_post_encode_ci_fails(self, tmp_path):
        response = Mock()
        response.text = (
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        )
        response.duration_ms = 1
        response.tokens = None
        response.estimated_cost_usd = None
        response.actual_cost_usd = None
        response.trace = {}
        response.unexpected_accesses = []
        response.error = None

        with (
            patch("axiom_encode.harness.evals._run_prompt_eval", return_value=response),
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
                return_value=EvalArtifactMetrics(
                    compile_pass=True,
                    compile_issues=[],
                    ci_pass=False,
                    ci_issues=["missing corpus source verification"],
                    embedded_source_present=True,
                    grounded_numeric_count=0,
                    ungrounded_numeric_count=0,
                    grounding=[],
                ),
            ),
        ):
            [result] = run_source_eval(
                source_id="sample",
                source_text="source",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=tmp_path / "rulespec-us",
                mode="cold",
            )

        assert result.success is False
        assert result.error == "Generated RuleSpec failed CI validation"


class TestCodexPromptEval:
    def test_wait_for_codex_process_terminates_after_stable_last_message(
        self, tmp_path
    ):
        last_message = tmp_path / ".codex-last-message.txt"
        last_message.write_text("ready\n")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        terminated = _wait_for_codex_process(
            process,
            last_message,
            timeout=1,
            settle_seconds=0,
            poll_interval=0,
        )

        assert terminated is True
        assert process.terminated is True


def test_build_eval_prompt_targets_rulespec_yaml(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="tn_snap_standard_utility_allowance",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="The standard utility allowance is $451.",
        axiom_rules_path=tmp_path / "us-tn",
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "tn_snap_standard_utility_allowance",
        "cold",
        workspace,
        [],
        target_file_name="tn-snap-standard-utility-allowance.yaml",
        include_tests=True,
        policyengine_rule_hint="snap_standard_utility_allowance",
    )

    assert "format: rulespec/v1" in prompt
    assert "RuleSpec YAML" in prompt
    assert "=== FILE: tn-snap-standard-utility-allowance.yaml ===" in prompt
    assert "=== FILE: tn-snap-standard-utility-allowance.test.yaml ===" in prompt
    assert "apply that limit at the source-stated lower entity" in prompt
    assert "then cap the aggregate" in prompt
    assert "rate-applied result at the source-stated lower entity" in prompt
    assert "unit-level placeholder or aggregate base by the rate" in prompt
    assert '"per taxpayer per beneficiary"' in prompt
    assert "Do not apply one\n  per-unit cap to a single aggregate amount" in prompt
    assert "For claim, overpayment, overissuance, repayment" in prompt
    assert "as collectability caps" in prompt
    assert "not return a bare placeholder such as `claim_amount`" in prompt
    assert (
        "Imported definitions do not override the current source's legal subject"
        in prompt
    )
    assert "rate-applied result at that lower entity" in prompt
    assert "keep the result executable by exposing a\n  local boundary input" in prompt
    assert "such as `wages`" in prompt
    assert 'contextual qualifiers such as\n  "received by"' in prompt
    assert "do not fold them into the boundary input name" in prompt
    assert "Treat legal subject nouns as stronger evidence" in prompt
    assert "use `entity: Person` for the current source's own amount" in prompt
    assert (
        "Existing target or repository-precedent files are not entity-scope authority"
        in prompt
    )
    assert "treat the copied aggregate shape as a defect to repair" in prompt
    assert (
        "Do not preserve the\n  aggregate entity just to keep old output names or tests compatible"
        in prompt
    )
    assert "earned income of an individual shall be\n  computed" in prompt
    assert "replaced by one aggregated boundary input" in prompt
    assert "thresholded, capped, base-limited" in prompt
    assert (
        "do not flatten the cited mechanics into `current_base * imported_rate`"
        in prompt
    )
    assert "defer the\n  affected executable output" in prompt
    assert 'definition uses "taxpayer" but also says the amount is "of an' in prompt
    assert 'Do not let\n  the word "taxpayer"' in prompt
    assert "on the [base] of every individual/person/employee" in prompt
    assert "even if the imported base definition or its tests are unit-scoped" in prompt
    assert "Do not narrate your plan" in prompt
    assert "snap_standard_utility_allowance" in prompt
    assert "Do not use bare year periods like `2024`" in prompt
    assert "never use `period_kind: calendar_year`" in prompt
    assert "period_kind: tax_year" in prompt
    assert "period_kind: custom" in prompt
    assert "period: Day" in prompt
    assert "never use bare `YYYY-MM-DD` shorthand" in prompt
    assert "Do not preserve existing `#input.filing_status`" in prompt
    assert 'If the source says only "joint return"' in prompt
    assert 'status 4 falls under any "other case" branch' in prompt
    assert "Existing executable output names are public API contracts" not in prompt
    assert "applicable_amount_in_effect_under_section_<section>" not in prompt
    assert "Do not put the date or year value in the fact name" in prompt
    assert "Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`" in prompt
    assert "overrides preservation of existing local input names" in prompt
    assert "Never introduce an import cycle" in prompt
    assert (
        "Never create a derived rule whose formula references that same rule's name"
        in prompt
    )
    assert "no local derived rule formula references its own\n      rule name" in prompt
    assert "directly or transitively" in prompt
    assert "numeric boundary input" in prompt
    assert "do not import that consumer section" in prompt
    assert "purpose-specific outputs such as `x_for_section_1234_a`" in prompt
    assert "purpose-specific branch into one generic output" in prompt
    assert "same-named local input such as `x`" in prompt
    assert "export matching the requested source's clause" in prompt
    assert "trailing commas in calls" in prompt
    assert "do not assume one upstream raw input equals that imported output" in prompt
    assert "For IRC section 151 repairs" not in prompt
    assert "named numeric concept" in prompt
    assert "1 / benefit_cost_rate_compensation_lookback_years" in prompt
    assert "`1 / 5`" in prompt
    assert "if the source is a multi-state or\n  multi-jurisdiction table" in prompt
    assert "Do not invent a fake `State` entity" in prompt
    assert "do not create one scalar parameter per row, bound, or cell" in prompt
    assert "`*_lower_bound_band_9`" in prompt
    assert "`indexed_by: <band_selector>`" in prompt
    assert "integer band ids such as `0`, `1`, and `2`" in prompt
    assert "do not use decimal row thresholds like `1.33`, `2.5`" in prompt
    assert "or strings such as `2_5_to_less_than_3_0`" in prompt
    assert "For interval-table repair of an existing target" in prompt
    assert "do not add extra exported derived rules" in prompt
    assert "`clause_ii_provides_otherwise`" in prompt
    assert "Do not treat the final interval row as open-ended" in prompt
    assert "Include a companion test above the final bounded row" in prompt
    assert "The out-of-table sentinel is not itself a source table row" in prompt
    assert "do not clamp sentinel cases" in prompt
    assert "Use a negative sentinel such as `-1`" in prompt
    assert "do not use the next positive band id such as `6`" in prompt
    assert "Do not hard-code the final real band id" in prompt
    assert "let the indexed interpolation formula produce that constant" in prompt
    assert "source text `133%` should be represented as `1.33`" in prompt
    assert "old percent-point test inputs" in prompt
    assert "Structural interval bounds that are only used by the selector" in prompt
    assert "private bound concepts" in prompt
    assert "do not preserve, rename, or recreate the local" in prompt
    assert "for sibling clause\n  exception phrases" in prompt
    assert "Before finalizing, do this self-check:" in prompt
    assert "Numeric inventory: every source-stated legal amount" in prompt
    assert "exact imported concept\n     from context" in prompt
    assert "indexed numeric concepts" in prompt
    assert "import it instead of duplicating it locally" in prompt
    assert "An imported `dtype: Judgment` is a predicate, not a scalar" in prompt
    assert "Never multiply, add, subtract, divide, `min`, or `max`" in prompt
    assert "encode the source-stated numeric base as a local amount fact" in prompt
    assert "Test input inventory: for every local factual identifier" in prompt
    assert "For proration, average, ratio, or percentage tests" in prompt
    assert "use totals like 600" in prompt
    assert "Avoid exact equality boundaries for ratios or percentages" in prompt
    assert "Do not assert raw `kind: parameter` rules directly" in prompt
    assert "assert derived outputs that consume the parameters" in prompt
    assert "modifier parameter stranded" in prompt
    assert "module.deferred_outputs[]" in prompt
    assert "source_values" in prompt
    assert "in excess of" in prompt
    assert "max(0, measured_value - limit)" in prompt
    assert "Do not defer that excess output merely" in prompt
    assert "final effective legal amount" in prompt
    assert "explanatory percentage or index" in prompt
    assert "unused modifier parameter" in prompt
    assert "do not model that numeric term as a local" in prompt
    assert "tier_1_applicable_percentage" in prompt
    assert "output` target path must include that source path segment" in prompt
    assert "us:statutes/26/3201/a#tier_1_employee_tax" in prompt
    assert (
        "Only include `blocked_by` entries when you know the exact RuleSpec output"
        in prompt
    )
    assert "Do not list bare legal provisions" in prompt
    assert "us:statutes/us-ca/17000" in prompt
    assert "exclusions conditioned on a reasonable belief" in prompt
    assert "Do not defer solely because" in prompt
    assert "model the source-stated\n  reasonable-belief condition" in prompt
    assert "enumerates qualifying or exception categories" in prompt
    assert "cites other laws only to define those category labels" in prompt
    assert "only uses the citation to label a category" in prompt
    assert "appointment, office, retirement-system, election" in prompt
    assert "`described in section ...` category\n  labels" in prompt
    assert "section-described supporting organization" in prompt
    assert "covered-service, section-described supporting organization" in prompt
    assert "`within the\n  meaning of section ...` carve-outs" in prompt
    assert "category membership phrases" in prompt
    assert "`organization described in section X`" in prompt
    assert "organization_described_in_section_509_a_3" in prompt
    assert "testing\n  membership in the described category" in prompt
    assert (
        "unrelated-trade-or-business, or other\n  within-meaning/described-in definitions"
        in prompt
    )
    assert "Validation fails if a direct local `#input.*_exception_applies`" in prompt
    assert "imported test inputs from copied files" in prompt
    assert "Do not stub imported derived" in prompt
    assert "never assign prohibited derived" in prompt
    assert (
        "classifications such as any imported or local `#input.filing_status`" in prompt
    )
    assert "omit that assertion or encode the" in prompt
    assert "upstream filing-status" in prompt
    assert "sources first" in prompt
    assert "#relation.<name>` input value must be a YAML list of row mappings" in prompt
    assert "member_of_household: [- true]" in prompt
    assert "tables.<Entity>` rows" in prompt
    assert "Proof inventory: every proof atom uses only an allowed `kind`" in prompt
    assert (
        "Import inventory: every `imports:` entry is an exact copied/importable"
        in prompt
    )
    assert "Top-level `imports:` entries must be scalar strings" in prompt
    assert "map entries like `- target:`" in prompt
    assert (
        "Supported scalar functions are `min(...)`, `max(...)`, `floor(x)`, and `ceil(x)`"
        in prompt
    )
    assert "Do not use Python-only functions such as `round(...)`" in prompt
    assert (
        "Use `sum(relation.amount_fact)` only when `amount_fact` is a raw scalar fact"
        in prompt
    )
    assert "Do not use `sum(relation.local_output)`" in prompt
    assert "Do not write `amount + if condition: extra else: 0`" in prompt
    assert "Do not emit more than one `versions:` entry for `kind: derived`" in prompt
    assert (
        "A `kind: table_cell` proof atom must include `source.table.header`" in prompt
    )
    assert "header-only `parameter_table` proof atoms are invalid" in prompt
    assert "row_key" in prompt
    assert "column_key" in prompt
    assert "kind: derived_relation" in prompt
    assert "derived_relation:" in prompt
    assert "arity: 2" in prompt
    assert "source_relation: member_of_household" in prompt
    assert "formula: snap_member_eligible" in prompt
    assert "explicitly defines" in prompt
    assert "membership in a derived legal unit" in prompt
    assert '"This source is about SNAP" is not enough' in prompt
    assert "stay on the source-stated structural entity" in prompt
    assert "Any rule that uses `entity: <filtered-entity>`" in prompt
    assert "declare" in prompt
    assert "that entity with a `kind: derived_relation` rule" in prompt
    assert "import a RuleSpec file" in prompt
    assert "that declares it" in prompt
    assert "example_output\n    kind: derived\n    entity: Household" in prompt
    assert (
        "Adjacent bracket thresholds repeated as both an upper bound and the next"
        in prompt
    )


def test_build_eval_prompt_for_rate_only_source_id_limits_scope(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="us/statute/26/1401/a/rate",
        runner=runner,
        output_root=tmp_path / "out",
        source_text=(
            "(a) Old-age, survivors, and disability insurance There shall be "
            "imposed for each taxable year, on the self-employment income of "
            "every individual, a tax equal to 12.4 percent of the amount of "
            "the self-employment income for such taxable year."
        ),
        axiom_rules_path=tmp_path / "rulespec-us",
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "us/statute/26/1401/a/rate",
        "cold",
        workspace,
        [],
        target_file_name="rate.yaml",
        target_ref_prefix="us:statutes/26/1401/a/rate",
        include_tests=True,
    )

    assert "Rate-only source boundary:" in prompt
    assert "source-stated rate or percentage" in prompt
    assert "parameters anchored in `./source.txt`" in prompt
    assert "Do not encode the downstream tax" in prompt
    assert "Prefer `kind: parameter`, `dtype: Rate`" in prompt
    assert "boundary must stay acyclic" in prompt
    assert "companion tests may assert" in prompt
    assert "canonical parameter output directly" in prompt
    assert "Explicit rate-only source-boundary artifacts" in prompt


def test_target_source_scope_ignores_cross_references_before_structural_marker():
    source = "\n\n".join(
        [
            "(a) Sampling plan. The plan references paragraph (b)(4), paragraph "
            "(b)(1)(iii), and paragraph (b)(2)(ii) before the actual sample-size "
            "paragraph.",
            "(b) Sample size. The State agency shall review active and negative cases.",
            "(1) Active cases. (i) All active cases shall be selected.",
            "(ii) Unless the alternate active case formula applies, the sample size is:",
            "Average monthly reviewable caseload (N) | Minimum annual sample size (n)\n"
            "60,000 and over | n = 2400\n"
            "10,000 to 59,999 | n = 300 + [0.042(N-10,000)]\n"
            "Under 10,000 | n = 300",
            "(iii) A State agency with the certification may instead use 0.0153.",
            "(2) Negative cases. (i) Unless the State agency uses paragraph "
            "(b)(2)(ii), the negative sample size is:",
            "Average monthly reviewable negative caseload (N) | Minimum annual sample size (n)\n"
            "5,000 and over | n = 800\n"
            "500 to 4,999 | n = 150 + [0.144(N-500)]\n"
            "Under 500 | n = 150",
            "(ii) A State agency with the certification may determine the negative "
            "sample size as follows:",
            "Average monthly reviewable negative caseload (N) | Minimum annual sample size (n)\n"
            "5,000 and over | n = 680\n"
            "684 to 4,999 | n = 150 + [0.1224(N-683)]\n"
            "Under 684 | n = 150",
            "(iii) In the formulas, n is the required negative sample size.",
            "(c) Review process.",
        ]
    )

    regular_negative = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/11/b/2/i",
    )
    assert regular_negative.lstrip().startswith("(i) Unless")
    assert "0.144" in regular_negative
    assert "0.1224" not in regular_negative

    alternate_negative = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/11/b/2/ii",
    )
    assert alternate_negative.lstrip().startswith("(ii) A State agency")
    assert "0.1224" in alternate_negative
    assert "0.144" not in alternate_negative

    regular_active = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/11/b/1/ii",
    )
    assert regular_active.lstrip().startswith("(ii) Unless")
    assert "0.042" in regular_active
    assert "0.0153" not in regular_active


def test_target_source_scope_distinguishes_alpha_marker_case_by_level():
    source = "\n\n".join(
        [
            "(d) Validation of State Agency error rates.",
            "(1) Payment error rate. (i) FNS will select a subsample.",
            "(A) First active subsample formula.",
            "(B) Second active subsample formula.",
            "(E) N is the State agency's minimum active case sample size.",
            "(2) Other payment-error review steps.",
            "(3) Negative case error rate. (i) FNS will select a subsample of "
            "completed negative cases as follows:",
            "Average monthly reviewable negative caseload (N) | Federal subsample target (n')\n"
            "12,000 and over | n' = 400\n"
            "1,001 to 11,999 | n' = .011634 N + 40\n"
            "1,000 and under | n' = 150",
            "(ii) The negative case record review follows.",
            "(e) State corrective action.",
        ]
    )

    negative_review = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/3/d/3/i",
    )

    assert negative_review.lstrip().startswith("(i) FNS will select")
    assert "Federal subsample target" in negative_review
    assert "Second active subsample formula" not in negative_review
    assert "(e) State corrective action" not in negative_review


def test_target_source_scope_treats_uppercase_alpha_as_non_roman():
    source = "\n\n".join(
        [
            "(b) State agency error rates.",
            "(2) Determination of payment error rates.",
            "(i) FNS shall calculate regressed error rates.",
            "(A) y1' = y1 + b1 (X1 - x1).",
            "(B) y2' = y2 + b2 (X2 - x2).",
            "(C) The regressed error rates are r1' = y1'/u and r2' = y2'/u.",
            "(D) The adjusted regressed payment error rate is r1'' + r2''.",
            "(ii) Other review steps.",
        ]
    )

    regressed_rates = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/23/b/2/i/C",
    )

    assert regressed_rates.lstrip().startswith("(C) The regressed error rates")
    assert "r1' = y1'/u" in regressed_rates
    assert "adjusted regressed payment error rate" not in regressed_rates


def test_target_source_scope_prefers_line_start_nested_markers():
    source = "\n\n".join(
        [
            "(f) Good cause.",
            "(1) Natural disasters. (i) The State agency shall document impacts.",
            "(ii) (A) The following criteria apply:",
            "(1) Geographic impact;",
            "(2) Duration;",
            "(3) The proportion of caseload affected; and/or",
            "(4) Operational impact.",
            "(2) Strikes.",
            "(3) Caseload growth.",
            "(i) A State agency may request relief for unusual caseload growth.",
            "(ii) Criteria apply.",
            "(iii) If information is insufficient, use this five-step calculation:",
            "(A) Step 1--determine the base-period average.",
            "(B) Step 2--determine the percentage increase.",
            "(C) Step 3--determine the percentage the error rate exceeds the national performance measure.",
            "(D) Step 4--divide the percentage increase by the percentage excess.",
            "(E) Step 5--multiply the quotient by the liability amount.",
            "(iv) Caseload growth of less than 15 percent is not considered.",
            "(4) Program changes.",
            "(g) Results of appeals.",
        ]
    )

    step_three = _target_source_scope_for_heuristics(
        source,
        "us:regulations/7-cfr/275/23/f/3/iii/C",
    )

    assert step_three.lstrip().startswith("(C) Step 3")
    assert "percentage the error rate exceeds" in step_three
    assert "Step 4" not in step_three
    assert "The proportion of caseload affected" not in step_three


def test_build_eval_prompt_does_not_treat_rates_path_as_rate_only(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="us/statute/26/1401/rates",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="The table states several percentage rates.",
        axiom_rules_path=tmp_path / "rulespec-us",
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "us/statute/26/1401/rates",
        "cold",
        workspace,
        [],
        target_file_name="rates.yaml",
        target_ref_prefix="us:statutes/26/1401/rates",
        include_tests=True,
    )

    assert "Rate-only source boundary:" not in prompt


def test_build_eval_prompt_does_not_treat_monetary_rate_as_rate_only(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="us/statute/26/9999/rate",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="The reimbursement rate is 67 cents per mile.",
        axiom_rules_path=tmp_path / "rulespec-us",
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "us/statute/26/9999/rate",
        "cold",
        workspace,
        [],
        target_file_name="rate.yaml",
        target_ref_prefix="us:statutes/26/9999/rate",
        include_tests=True,
    )

    assert "Rate-only source boundary:" not in prompt


def test_context_file_surfaces_include_derived_relation(tmp_path):
    context_file = tmp_path / "snap_unit.yaml"
    context_file.write_text(
        """format: rulespec/v1
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
    )

    surfaces = _context_file_executable_surfaces(str(context_file))

    assert surfaces["snap_unit"]["kind"] == "derived_relation"
    assert surfaces["snap_unit"]["entity"] == "SnapUnit"


def test_build_eval_prompt_lists_existing_target_surfaces(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    target = policy_repo / "statutes/26/999.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(
        """format: rulespec/v1
rules:
  - name: existing_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          existing_fact
          and filing_status == 1
          and taxable_year_begins_after_2024
          and applicable_amount_in_effect_under_section_68_b > 0
  - name: existing_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 100
"""
    )
    workspace = prepare_eval_workspace(
        citation="26 USC 999",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="The amount is allowed.",
        axiom_rules_path=policy_repo,
        mode="repo-augmented",
        extra_context_paths=[],
    )
    cyclic_context = policy_repo / "statutes/26/7703.yaml"
    cyclic_context.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/999#existing_amount
rules:
  - name: upstream_married_rule
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: existing_amount > 0
"""
    )
    section_68_context = policy_repo / "statutes/26/68/b.yaml"
    section_68_context.parent.mkdir(parents=True)
    section_68_context.write_text(
        """format: rulespec/v1
rules:
  - name: section_68_applied_after_other_itemized_deduction_limitations
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: true
"""
    )
    workspace.context_files.append(
        EvalContextFile(
            source_path=str(cyclic_context),
            workspace_path="context/statutes/26/7703.yaml",
            import_path="us:statutes/26/7703",
            kind="citation_context",
        )
    )
    workspace.context_files.append(
        EvalContextFile(
            source_path=str(section_68_context),
            workspace_path="context/statutes/26/68/b.yaml",
            import_path="us:statutes/26/68/b",
            kind="citation_context",
        )
    )

    prompt = _build_eval_prompt(
        "26 USC 999",
        "repo-augmented",
        workspace,
        workspace.context_files,
        target_file_name="999.yaml",
        target_ref_prefix="us:statutes/26/999",
        include_tests=True,
        runner_backend="openai",
    )

    assert "Existing target executable surfaces:" in prompt
    assert "not compatibility contracts" in prompt
    assert "`us:statutes/26/999#existing_amount`" in prompt
    assert "entity=TaxUnit" in prompt
    assert "effective_from=2026-01-01" in prompt
    assert "`us:statutes/26/999#existing_table`" in prompt
    assert "indexed_by=household_size" in prompt
    assert "local input slots" in prompt
    assert "`us:statutes/26/999#input.existing_fact`" in prompt
    assert "Never copy a `#input` key from" in prompt
    assert "sibling context test" in prompt
    assert "Invalid copied local input names:" in prompt
    assert "`us:statutes/26/999#input.filing_status`" in prompt
    assert "filing status is a derived legal classification" in prompt
    assert "`us:statutes/26/999#input.taxable_year_begins_after_2024`" in prompt
    assert "date/year-valued temporal fact" in prompt
    assert "`post_YYYY`, `pre_YYYY`, or any four-digit year" in prompt
    assert (
        "`us:statutes/26/999#input.applicable_amount_in_effect_under_section_68_b`"
        in prompt
    )
    assert "encoded cross-reference placeholder" in prompt
    assert "us:statutes/26/68/b" in prompt
    assert "Existing target local factual inputs:" in prompt
    assert "`us:statutes/26/999#input.existing_fact`" in prompt
    assert "Cycle-prone context imports:" in prompt
    assert "`us:statutes/26/7703` already imports `us:statutes/26/999`" in prompt
    assert "more specific child file under the current target path" in prompt
    assert "such as a `/rate`" in prompt
    assert "do not emit a duplicate local `parameter`" in prompt


def test_build_eval_prompt_lists_existing_target_validation_failures(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    target = policy_repo / "statutes/26/63/f.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(
        """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
"""
    )
    workspace = prepare_eval_workspace(
        citation="26 USC 63(f)",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="The additional amount is $600. Substitute $750 for $600.",
        axiom_rules_path=policy_repo,
        mode="repo-augmented",
        extra_context_paths=[],
    )

    prompt = _build_eval_prompt(
        "26 USC 63(f)",
        "repo-augmented",
        workspace,
        workspace.context_files,
        target_file_name="f.yaml",
        target_ref_prefix="us:statutes/26/63/f",
        include_tests=True,
        runner_backend="openai",
    )

    assert "Copied existing target fails current RuleSpec validation:" in prompt
    assert "`us:statutes/26/63/f`" in prompt
    assert "unmarried_not_surviving_spouse_additional_amount" in prompt
    assert "`module.deferred_outputs[].source_values`" in prompt
    assert "preserve the failing shape" in prompt


def test_materialize_eval_artifact_writes_rulespec_bundle(tmp_path):
    output_file = tmp_path / "runner" / "source" / "tn-snap.yaml"
    llm_response = """=== FILE: tn-snap.yaml ===
format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
rules:
  - name: snap_standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
=== FILE: tn-snap.test.yaml ===
- name: base
  period: 2024-01
  input: {}
  output:
    snap_standard_utility_allowance: 451
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="The standard utility allowance is $451.",
    )

    assert wrote is True
    assert output_file.exists()
    assert output_file.with_name("tn-snap.test.yaml").exists()
    assert output_file.read_text().startswith("format: rulespec/v1")


def test_materialize_eval_artifact_repairs_copied_cross_reference_summary(tmp_path):
    output_file = tmp_path / "runner" / "statutes" / "39" / "39-22-104" / "1.5.yaml"
    llm_response = """=== FILE: 1.5.yaml ===
format: rulespec/v1
module:
  summary: |-
    (1.5) Subject to subsection (2) of this section, a tax of four and three-quarters percent is imposed.

    (2) Prior to the application of the rate of tax prescribed in subsection (1), (1.5), or (1.7) of this section, federal taxable income shall be modified.
rules:
  - name: individual_estate_trust_income_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1999-01-01'
        formula: '0.0475'
"""

    wrote = _materialize_eval_artifact(llm_response, output_file)

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    summary = payload["module"]["summary"]
    assert "four and three-quarters percent" in summary
    assert "Prior to the application" not in summary


def test_rulespec_validation_overlay_preserves_eval_source_metadata(tmp_path):
    policy_repo = tmp_path / "rulespec-us-co"
    policy_repo.mkdir()
    output_file = (
        tmp_path
        / "out"
        / "openai-gpt-5.5"
        / "statutes"
        / "39"
        / "39-22-104"
        / "1.5.yaml"
    )
    output_file.parent.mkdir(parents=True)
    output_file.write_text("format: rulespec/v1\nrules: []\n")
    workspace = (
        tmp_path
        / "out"
        / "_eval_workspaces"
        / "openai-gpt-5.5"
        / "us-co-statute-39-39-22-104-1.5"
        / "workspace"
    )
    workspace.mkdir(parents=True)
    (workspace / "context-manifest.json").write_text(
        json.dumps(
            {
                "citation": "us-co/statute/39/39-22-104/1.5",
                "source_metadata": {
                    "corpus_citation_path": "us-co/statute/39/39-22-104",
                    "requested_source": "us-co/statute/39/39-22-104/1.5",
                },
            }
        )
    )

    with _rulespec_validation_target(output_file, policy_repo) as validation_file:
        metadata = validator_pipeline._load_nearby_eval_source_metadata(validation_file)

    assert metadata is not None
    assert metadata["requested_source"] == "us-co/statute/39/39-22-104/1.5"


def test_materialize_eval_artifact_repairs_source_table_band_scalars(tmp_path):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines applicable percentages by benefits ratio.
rules:
  - name: average_account_benefits_ratio_lower_bound_band_0
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_upper_bound_band_0
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_lower_bound_band_0:
            -1
          elif average_account_benefits_ratio < average_account_benefits_ratio_upper_bound_band_0:
            0
          else:
            1
  - name: applicable_percentage_3201_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.049
          1: 0
=== FILE: b.test.yaml ===
- name: selector_band
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/3241/b#input.average_account_benefits_ratio: 2.75
  output:
    us:statutes/26/3241/b#average_account_benefits_ratio_lower_bound_band_0: 2.5
    us:statutes/26/3241/b#applicable_percentage_3201_by_average_account_benefits_ratio_band:
      0: 0.049
      1: 0
    us:statutes/26/3241/b#average_account_benefits_ratio_band: 0
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Tax rate schedule | Average account benefits ratio | 2.5 | 3.0",
    )

    assert wrote is True
    content = output_file.read_text()
    assert "average_account_benefits_ratio_lower_bound_band_0" in content
    assert (
        "average_account_benefits_ratio < "
        "average_account_benefits_ratio_lower_bound_band_0"
    ) in content
    assert "elif" not in content
    assert "else if" not in content
    test_content = output_file.with_name("b.test.yaml").read_text()
    assert "average_account_benefits_ratio_lower_bound_band_0" in test_content
    assert (
        "applicable_percentage_3201_by_average_account_benefits_ratio_band"
        not in test_content
    )
    assert "average_account_benefits_ratio_band" in test_content


def test_materialize_eval_artifact_repairs_named_band_threshold_scalars(tmp_path):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines applicable percentages by benefits ratio.
rules:
  - name: average_account_benefits_ratio_band_threshold_2_5
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_threshold_3_0
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_band_threshold_2_5: 1
          else: if average_account_benefits_ratio < average_account_benefits_ratio_band_threshold_3_0: 2
          else: 3
  - name: applicable_percentage_3201_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.049
          2: 0.049
          3: 0
=== FILE: b.test.yaml ===
- name: selector_band
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/3241/b#input.average_account_benefits_ratio: 2.75
  output:
    us:statutes/26/3241/b#average_account_benefits_ratio_band_threshold_2_5: 2.5
    us:statutes/26/3241/b#applicable_percentage_3201_by_average_account_benefits_ratio_band:
      1: 0.049
      2: 0.049
      3: 0
    us:statutes/26/3241/b#average_account_benefits_ratio_band: 2
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Tax rate schedule | Average account benefits ratio | 2.5 | 3.0",
    )

    assert wrote is True
    content = output_file.read_text()
    assert "average_account_benefits_ratio_band_threshold_2_5" in content
    assert "average_account_benefits_ratio_band_threshold_3_0" in content
    assert (
        "average_account_benefits_ratio < "
        "average_account_benefits_ratio_band_threshold_2_5"
    ) in content
    assert (
        "average_account_benefits_ratio < "
        "average_account_benefits_ratio_band_threshold_3_0"
    ) in content
    test_content = output_file.with_name("b.test.yaml").read_text()
    assert "average_account_benefits_ratio_band_threshold_2_5" in test_content
    assert (
        "applicable_percentage_3201_by_average_account_benefits_ratio_band"
        not in test_content
    )
    assert "average_account_benefits_ratio_band" in test_content


def test_materialize_eval_artifact_repairs_chained_conditionals_without_table_scalars(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines applicable percentages by benefits ratio.
rules:
  - name: average_account_benefits_ratio_at_least_threshold_by_band
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 2.5
          2: 3.0
  - name: average_account_benefits_ratio_but_less_than_threshold_by_band
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_band
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              corpus_citation_path: us/statute/26/3241
              table:
                header: Tax rate schedule
                row_key: Average account benefits ratio band
                column_key: But less than
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 3.0
          2: 1000000.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_at_least_threshold_by_band[1]:
            0
          else if average_account_benefits_ratio >= average_account_benefits_ratio_at_least_threshold_by_band[2]:
            2
          else:
            1
=== FILE: b.test.yaml ===
- name: selector_band
  period: 2026
  input:
    average_account_benefits_ratio: 2.75
  output:
    average_account_benefits_ratio_band: 1
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Tax rate schedule | Average account benefits ratio | 2.5 | 3.0",
    )

    assert wrote is True
    content = output_file.read_text()
    assert "elif" not in content
    assert "else if" not in content
    assert "1000000.0" not in content
    payload = yaml.safe_load(content)
    assert 2 not in payload["rules"][1]["versions"][0]["values"]
    formula = payload["rules"][2]["versions"][0]["formula"]
    assert formula.startswith("if average_account_benefits_ratio < ")
    assert " else: if average_account_benefits_ratio >= " in formula


def test_materialize_eval_artifact_repairs_multiline_conditional_branch(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "1402" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines self-employment income.
rules:
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if self_employment_income_excluded_as_nonresident_alien:
            0
          else if net_earnings_from_self_employment < minimum_self_employment_income_threshold:
            0
          else:
            min(
              max(0, net_earnings_from_self_employment),
              max(
                0,
                contribution_and_benefit_base_under_section_230_of_social_security_act
                  - wages_paid_to_individual_during_taxable_year
              )
            )
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="The OASDI base is reduced by wages paid during the year.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert "else if" not in formula
    assert "elif" not in formula
    assert (
        "else: if net_earnings_from_self_employment "
        "< minimum_self_employment_income_threshold: 0 else: min("
    ) in formula
    assert (
        "contribution_and_benefit_base_under_section_230_of_social_security_act"
        in formula
    )


def test_materialize_eval_artifact_repairs_multiline_else_if_conditions(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "1402" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines self-employment income.
rules:
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if is_nonresident_alien_individual_for_chapter_1402:
            0
          else if church_employee_income
              and apply_section_1402_j_2_special_rules_for_church_income:
            self_employment_income_for_church_employee_under_section_1402_j_2
          else if net_earnings_from_self_employment
              < self_employment_income_small_amount_exclusion_threshold:
            0
          else:
            net_earnings_from_self_employment
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="The special rules apply in the case of church employee income.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert "else if" not in formula
    assert "elif" not in formula
    assert (
        "if church_employee_income and "
        "apply_section_1402_j_2_special_rules_for_church_income: "
        "self_employment_income_for_church_employee_under_section_1402_j_2"
    ) in formula
    assert (
        "if net_earnings_from_self_employment "
        "< self_employment_income_small_amount_exclusion_threshold: 0"
    ) in formula


def test_materialize_eval_artifact_repairs_then_conditionals(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "1402" / "b.yaml"
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: Section defines self-employment income.
rules:
  - name: self_employment_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if net_earnings_from_self_employment >= self_employment_income_minimum_amount
              and not individual_is_nonresident_alien_individual
              then: net_earnings_from_self_employment else: 0
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="Self-employment income excludes net earnings below $400.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert "then:" not in formula
    assert (
        "if net_earnings_from_self_employment >= self_employment_income_minimum_amount "
        "and not individual_is_nonresident_alien_individual: "
        "net_earnings_from_self_employment else: 0"
    ) == formula


def test_materialize_eval_artifact_repairs_python_ternary_formulas(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "164" / "f.yaml"
    llm_response = """=== FILE: f.yaml ===
format: rulespec/v1
module:
  summary: Section allows a self-employment tax deduction.
rules:
  - name: self_employment_tax_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          self_employment_tax_deduction_fraction * (
            old_age_survivors_and_disability_insurance_tax
            + self_employment_income_tax
          ) if taxpayer_is_individual else 0
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="An individual may deduct one-half of self-employment taxes.",
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    formula = payload["rules"][0]["versions"][0]["formula"]
    assert (
        "if taxpayer_is_individual: self_employment_tax_deduction_fraction * "
        "( old_age_survivors_and_disability_insurance_tax + "
        "self_employment_income_tax ) else: 0"
    ) == formula


def test_materialize_eval_artifact_preserves_open_interval_source_table_rows(
    tmp_path,
):
    output_file = tmp_path / "runner" / "statutes" / "26" / "3241" / "b.yaml"
    source_text = """Tax rate schedule | Average account benefits ratio | Applicable percentage for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
| At least | But less than | | | .............. | 2.5 | 22.1 | 4.9 |
| 2.5 | 6.1 | 18.1 | 4.9 |
| 6.1 | 9.0 | 12.6 | 4.4 |
| 9.0 | .............. | 8.2 | 0 |"""
    llm_response = """=== FILE: b.yaml ===
format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
    | At least | But less than | | | .............. | 2.5 | 22.1 | 4.9 |
    | 2.5 | 6.1 | 18.1 | 4.9 |
    | 6.1 | 9.0 | 12.6 | 4.4 |
    | 9.0 | .............. | 8.2 | 0 |
rules:
  - name: section_3241b_average_account_benefits_ratio_bracket_lower_bound
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_bracket
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              table:
                header: Tax rate schedule
                row_key: average account benefits ratio band
                column_key: At least
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 2.5
          2: 6.1
          3: 9.0
  - name: section_3241b_average_account_benefits_ratio_bracket_upper_bound
    kind: parameter
    dtype: Decimal
    indexed_by: average_account_benefits_ratio_bracket
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              table:
                header: Tax rate schedule
                row_key: average account benefits ratio band
                column_key: But less than
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 6.1
          2: 9.0
  - name: average_account_benefits_ratio_bracket
    kind: derived
    entity: TaxUnit
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio >= section_3241b_average_account_benefits_ratio_bracket_lower_bound[1] and average_account_benefits_ratio < section_3241b_average_account_benefits_ratio_bracket_upper_bound[1]:
            1
          else if average_account_benefits_ratio >= section_3241b_average_account_benefits_ratio_bracket_lower_bound[2] and average_account_benefits_ratio < section_3241b_average_account_benefits_ratio_bracket_upper_bound[2]:
            2
          else:
            0
  - name: section_3241b_sections_3211b_and_3221b_applicable_percentage_points
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0
          1: 0.221
          2: 0.181
          3: 0.126
  - name: section_3241b_section_3201b_applicable_percentage_points
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0
          1: 0.049
          2: 0.049
          3: 0.044
  - name: section_3211b_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: section_3241b_sections_3211b_and_3221b_applicable_percentage_points[average_account_benefits_ratio_bracket]
  - name: section_3201b_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: section_3241b_section_3201b_applicable_percentage_points[average_account_benefits_ratio_bracket]
=== FILE: b.test.yaml ===
- name: below_open_lower_row
  period: 2026
  input:
    average_account_benefits_ratio: 2.49
  output:
    average_account_benefits_ratio_bracket: 0
    section_3211b_applicable_percentage: 0.0
    section_3201b_applicable_percentage: 0.0
- name: boundary_2_5_uses_second_source_row
  period: 2026
  input:
    average_account_benefits_ratio: 2.5
  output:
    section_3211b_applicable_percentage: 0.221
    section_3201b_applicable_percentage: 0.049
- name: boundary_6_1_uses_next_source_row
  period: 2026
  input:
    average_account_benefits_ratio: 6.1
  output:
    section_3211b_applicable_percentage: 0.181
    section_3201b_applicable_percentage: 0.049
- name: open_upper_row
  period: 2026
  input:
    average_account_benefits_ratio: 9.0
  output:
    section_3211b_applicable_percentage: 0.126
    section_3201b_applicable_percentage: 0.044
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text=source_text,
    )

    assert wrote is True
    payload = yaml.safe_load(output_file.read_text())
    rule_names = {rule["name"] for rule in payload["rules"]}
    assert (
        "section_3241b_average_account_benefits_ratio_bracket_lower_bound" in rule_names
    )
    assert (
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound" in rule_names
    )
    lower_bound = next(
        rule
        for rule in payload["rules"]
        if rule["name"]
        == "section_3241b_average_account_benefits_ratio_bracket_lower_bound"
    )
    upper_bound = next(
        rule
        for rule in payload["rules"]
        if rule["name"]
        == "section_3241b_average_account_benefits_ratio_bracket_upper_bound"
    )
    assert lower_bound["versions"][0]["values"] == {2: 2.5, 3: 6.1, 4: 9.0}
    assert upper_bound["versions"][0]["values"] == {1: 2.5, 2: 6.1, 3: 9.0}
    selector = next(
        rule
        for rule in payload["rules"]
        if rule["name"] == "average_account_benefits_ratio_bracket"
    )
    assert selector["versions"][0]["formula"] == (
        "if average_account_benefits_ratio < "
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound[1]: 1 else: "
        "if average_account_benefits_ratio >= "
        "section_3241b_average_account_benefits_ratio_bracket_lower_bound[2] and "
        "average_account_benefits_ratio < "
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound[2]: 2 else: "
        "if average_account_benefits_ratio >= "
        "section_3241b_average_account_benefits_ratio_bracket_lower_bound[3] and "
        "average_account_benefits_ratio < "
        "section_3241b_average_account_benefits_ratio_bracket_upper_bound[3]: 3 else: 4"
    )
    rates = next(
        rule
        for rule in payload["rules"]
        if rule["name"]
        == "section_3241b_sections_3211b_and_3221b_applicable_percentage_points"
    )
    assert rates["versions"][0]["values"] == {
        1: 0.221,
        2: 0.181,
        3: 0.126,
        4: 0.082,
    }
    cases = {
        case["name"]: case
        for case in yaml.safe_load(output_file.with_name("b.test.yaml").read_text())
    }
    assert (
        cases["below_open_lower_row"]["output"][
            "average_account_benefits_ratio_bracket"
        ]
        == 1
    )
    assert (
        cases["below_open_lower_row"]["output"]["section_3211b_applicable_percentage"]
        == 0.221
    )
    assert (
        cases["boundary_2_5_uses_second_source_row"]["output"][
            "section_3211b_applicable_percentage"
        ]
        == 0.181
    )
    assert (
        cases["boundary_6_1_uses_next_source_row"]["output"][
            "section_3211b_applicable_percentage"
        ]
        == 0.126
    )
    assert (
        cases["open_upper_row"]["output"]["section_3211b_applicable_percentage"]
        == 0.082
    )


def test_run_source_eval_retries_once_when_first_response_has_no_rulespec(tmp_path):
    policy_repo_root = tmp_path / "axiom-rules-engine"
    policy_repo_root.mkdir()
    first_response = EvalPromptResponse(
        text="I'm going to encode a compact source-faithful slice.",
        duration_ms=10,
        trace={"attempt": "initial"},
    )
    second_response = EvalPromptResponse(
        text=(
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=20,
        trace={"attempt": "retry"},
    )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=[first_response, second_response],
        ) as mock_prompt_eval,
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_id="sample",
            source_text="source states 451.",
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            mode="cold",
        )

    assert result.success is True
    assert result.retry_count == 1
    assert result.duration_ms == 30
    assert Path(result.output_file).exists()
    assert mock_prompt_eval.call_count == 2
    retry_prompt = mock_prompt_eval.call_args_list[1].args[2]
    assert "previous response did not contain a RuleSpec artifact" in retry_prompt
    assert "Do not narrate your plan" in retry_prompt


def test_empty_artifact_retry_prompt_uses_minimal_source_scope_protocol():
    from axiom_encode.prompts.encoder import SOURCE_SCOPE_PROTOCOL

    original_prompt = (
        "Task preface.\n"
        f"{SOURCE_SCOPE_PROTOCOL}\n"
        "Additional encoding guidance:\n"
        "- Keep this instruction.\n"
    )

    retry_prompt = _build_empty_artifact_retry_prompt(
        original_prompt,
        target_file_name="sample.yaml",
        include_tests=True,
    )

    assert "Source-scope protocol (minimal):" in retry_prompt
    assert "Additional encoding guidance:" in retry_prompt
    assert "- Keep this instruction." in retry_prompt
    assert "do not promote it to a household, unit" not in retry_prompt
    assert "Return exactly this two-file bundle" in retry_prompt


def test_run_source_eval_retries_once_when_first_response_times_out(tmp_path):
    policy_repo_root = tmp_path / "axiom-rules-engine"
    policy_repo_root.mkdir()
    first_response = EvalPromptResponse(
        text="",
        duration_ms=300000,
        trace={"timed_out": True, "timeout_reason": "idle"},
        error="Codex eval timed out",
    )
    second_response = EvalPromptResponse(
        text=(
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=20,
        trace={"attempt": "retry"},
    )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=[first_response, second_response],
        ) as mock_prompt_eval,
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_id="sample",
            source_text="source states 451.",
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            mode="cold",
        )

    assert result.success is True
    assert result.retry_count == 1
    assert result.error is None
    assert result.duration_ms == 300020
    assert Path(result.output_file).exists()
    assert mock_prompt_eval.call_count == 2


def test_codex_prompt_timeouts_use_default_for_short_source(tmp_path):
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2012",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="short source",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (600, 300)


def test_codex_prompt_timeouts_use_env_for_short_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_IDLE_TIMEOUT_SECONDS", "30")
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2012",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="short source",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (90, 30)


def test_codex_prompt_timeouts_ignore_invalid_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_IDLE_TIMEOUT_SECONDS", "0")
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2012",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="short source",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (600, 300)


def test_codex_prompt_timeouts_use_long_limits_for_large_source(tmp_path):
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2014",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="x" * 40000,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (1800, 900)


def test_codex_prompt_timeouts_use_long_env_for_large_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_LONG_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("AXIOM_ENCODE_CODEX_LONG_IDLE_TIMEOUT_SECONDS", "60")
    workspace = prepare_eval_workspace(
        citation="us/statute/7/2014",
        runner=parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="x" * 40000,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        mode="cold",
        extra_context_paths=[],
    )

    assert _codex_prompt_timeouts(workspace) == (240, 60)


def test_run_source_eval_does_not_retry_when_first_response_writes_rulespec(tmp_path):
    policy_repo_root = tmp_path / "axiom-rules-engine"
    policy_repo_root.mkdir()
    response = EvalPromptResponse(
        text=(
            "=== FILE: sample.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: source states 451.\n"
            "rules: []\n"
            "=== FILE: sample.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=10,
        trace={"attempt": "initial"},
    )

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            return_value=response,
        ) as mock_prompt_eval,
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
    ):
        [result] = run_source_eval(
            source_id="sample",
            source_text="source states 451.",
            runner_specs=["codex:gpt-5.4"],
            output_root=tmp_path / "out",
            policy_path=policy_repo_root,
            mode="cold",
        )

    assert result.success is True
    assert result.retry_count == 0
    assert mock_prompt_eval.call_count == 1


def test_eval_result_payload_round_trips_prompt_digests():
    result = EvalResult(
        citation="snap_test",
        runner="codex-gpt-5.4",
        backend="codex",
        model="gpt-5.4",
        mode="repo-augmented",
        output_file="/tmp/snap_test.yaml",
        trace_file="/tmp/snap_test.trace.json",
        context_manifest_file="/tmp/snap_test.context.json",
        duration_ms=1234,
        success=True,
        error=None,
        input_tokens=11,
        output_tokens=22,
        cache_read_tokens=33,
        cache_creation_tokens=44,
        reasoning_output_tokens=55,
        estimated_cost_usd=0.12,
        actual_cost_usd=None,
        retrieved_files=["/tmp/context.yaml"],
        unexpected_accesses=[],
        retry_count=1,
        metrics=EvalArtifactMetrics(
            compile_pass=True,
            compile_issues=[],
            ci_pass=True,
            ci_issues=[],
            embedded_source_present=True,
            grounded_numeric_count=1,
            ungrounded_numeric_count=0,
            grounding=[],
            generalist_review_pass=True,
            generalist_review_score=9.0,
            generalist_review_issues=[],
            generalist_review_prompt_sha256="review-digest",
            policyengine_pass=True,
            policyengine_score=1.0,
            policyengine_issues=[],
            taxsim_pass=None,
            taxsim_score=None,
            taxsim_issues=[],
        ),
        generation_prompt_sha256="generation-digest",
    )

    restored = _eval_result_from_payload(result.to_dict())

    assert restored.generation_prompt_sha256 == "generation-digest"
    assert restored.retry_count == 1
    assert restored.metrics is not None
    assert restored.metrics.generalist_review_prompt_sha256 == "review-digest"

    def test_wait_for_codex_process_terminates_after_persistent_output(self, tmp_path):
        last_message = tmp_path / ".codex-last-message.txt"
        last_message.write_text("ready\n")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                if self.returncode is None:
                    last_message.touch()
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        terminated = _wait_for_codex_process(
            process,
            last_message,
            timeout=1,
            settle_seconds=1,
            max_output_wait_seconds=0,
            poll_interval=0,
        )

        assert terminated is True
        assert process.terminated is True

    def test_wait_for_codex_process_times_out_when_heartbeat_stalls(self, tmp_path):
        last_message = tmp_path / ".codex-last-message.txt"
        stdout_path = tmp_path / "stdout.log"
        stdout_path.write_text("")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        with pytest.raises(subprocess.TimeoutExpired):
            _wait_for_codex_process(
                process,
                last_message,
                timeout=1,
                heartbeat_paths=[stdout_path],
                max_idle_seconds=0,
                poll_interval=0,
            )

        assert process.terminated is True

    def test_run_codex_prompt_eval_accepts_stable_last_message_on_termination(
        self, tmp_path
    ):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/3/a",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="nil amount",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"
        event_lines = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "fallback"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "cached_input_tokens": 0,
                        },
                    }
                ),
            ]
        )

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd):
                self.args = cmd
                self.returncode = None
                Path(cwd, ".codex-last-message.txt").write_text(bundle)
                stdout.write(event_lines + "\n")
                stdout.flush()

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        def fake_wait(
            process,
            last_message_file,
            timeout,
            heartbeat_paths=None,
            settle_seconds=5.0,
            max_output_wait_seconds=30.0,
            max_idle_seconds=120.0,
            poll_interval=0.5,
        ):
            process.terminate()
            process.wait()
            return True

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=fake_wait,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is None
        assert response.text == bundle.strip()
        assert response.tokens is not None
        assert response.tokens.input_tokens == 10
        assert response.tokens.output_tokens == 4

    def test_run_codex_prompt_eval_salvages_last_message_on_timeout(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/paragraph/4A/1",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="maximum disregard",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd):
                self.args = cmd
                self.returncode = None
                Path(cwd, ".codex-last-message.txt").write_text(bundle)

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["codex", "exec"], timeout=600
                ),
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is None
        assert response.text == bundle.strip()

    def test_run_codex_prompt_eval_uses_longer_idle_timeout(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="10-ccr-2506-1/4.403.11/b/c/3",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="self-employment expenses",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd):
                self.args = cmd
                self.returncode = 0

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        observed: dict[str, float] = {}

        def fake_wait(
            process,
            last_message_file,
            timeout,
            heartbeat_paths=None,
            settle_seconds=5.0,
            max_output_wait_seconds=30.0,
            max_idle_seconds=120.0,
            poll_interval=0.5,
        ):
            observed["max_idle_seconds"] = max_idle_seconds
            process.returncode = 0
            return False

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=fake_wait,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert observed["max_idle_seconds"] == 300
        assert response.text == ""


class TestEvaluateArtifact:
    def test_validates_generated_artifact_inside_policy_repo_overlay(self, tmp_path):
        policy_repo = tmp_path / "repos" / "rulespec-us-ny"
        policy_repo.mkdir(parents=True)
        generated = (
            tmp_path
            / "out"
            / "openai-gpt-5.5"
            / "regulations"
            / "18-nycrr"
            / "387"
            / "12"
            / "f"
            / "3"
            / "v"
            / "c.yaml"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        generated.with_name("c.test.yaml").write_text("[]\n")
        seen_targets: list[tuple[tuple[str, ...], str, bool]] = []
        seen_policy_repo_roots: list[tuple[str, ...]] = []

        def fake_compile(_pipeline, path):
            seen_targets.append(
                (path.parts, path.name, path.with_name("c.test.yaml").exists())
            )
            seen_policy_repo_roots.append(_pipeline.policy_repo_path.parts)
            return ValidationResult("compile", passed=True)

        def fake_ci(_pipeline, path):
            seen_targets.append(
                (path.parts, path.name, path.with_name("c.test.yaml").exists())
            )
            seen_policy_repo_roots.append(_pipeline.policy_repo_path.parts)
            return ValidationResult("ci", passed=True)

        with (
            patch.object(ValidatorPipeline, "_run_compile_check", fake_compile),
            patch.object(ValidatorPipeline, "_run_ci", fake_ci),
        ):
            evaluate_artifact(
                rulespec_file=generated,
                policy_repo_root=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text="No numeric values.",
            )

        assert len(seen_targets) == 2
        for parts, name, companion_test_exists in seen_targets:
            assert "rulespec-us-ny" in parts
            assert name == "c.yaml"
            assert companion_test_exists
        for parts in seen_policy_repo_roots:
            assert "rulespec-us-ny" in parts
            assert parts != policy_repo.parts

    def test_uses_fallback_source_text_for_grounding(self, tmp_path):
        rulespec_file = tmp_path / "24" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: (a) Allowance of credit There shall be allowed a credit of $1,000.\n"
            "rules:\n"
            "  - name: ctc_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 1000\n"
            "      - effective_from: '2025-01-01'\n"
            "        formula: 2200\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)

        with (
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
                return_value=compile_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_ci",
                return_value=ci_result,
            ),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "(a) Allowance of credit There shall be allowed a credit of $1,000."
                ),
            )

        assert metrics.compile_pass
        assert not metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 1
        assert [item.raw for item in metrics.grounding if not item.grounded] == ["2200"]
        assert any(
            "Ungrounded generated numeric literal" in issue and "2200" in issue
            for issue in metrics.ci_issues
        )

    def test_evaluate_artifact_skips_reviewers_when_requested(self, tmp_path):
        rulespec_file = tmp_path / "24" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Source text says the amount is $1,000.\n"
            "rules:\n"
            "  - name: ctc_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 1000\n"
        )

        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", passed=True),
            ),
            patch.object(ValidatorPipeline, "_run_reviewer") as mock_reviewer,
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Source text says the amount is $1,000.",
                skip_reviewers=True,
            )

        mock_reviewer.assert_not_called()
        assert metrics.generalist_review_pass
        assert metrics.generalist_review_score is None
        assert metrics.generalist_review_issues == []

    def test_generated_eval_repairs_unreferenced_proof_imports(self, tmp_path):
        rulespec_file = tmp_path / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
imports:
  - us-co:regulations/example#deadline
rules:
  - name: result_rule
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              excerpt: missed deadline
          - path: versions[0].formula
            kind: import
            import:
              target: us-co:regulations/example#deadline
              output: deadline
              hash: sha256:local
    versions:
      - effective_from: '2025-01-01'
        formula: missed_deadline
"""
        )

        ci_issue = (
            "Proof import not referenced: `result_rule` proof imports `deadline`, "
            "but the rule formula does not reference that imported symbol."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The office missed the deadline.",
                skip_reviewers=True,
            )

        repaired_text = rulespec_file.read_text()
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert "kind: import" not in repaired_text
        assert "output: deadline" not in repaired_text

    def test_generated_eval_repairs_unused_imports(self, tmp_path):
        repo = tmp_path / "rulespec-us"
        rulespec_file = repo / "statutes" / "26" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/source#unused_rate
  - us:statutes/26/source#used_rate
rules:
  - name: copied_rate
    kind: derived
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: used_rate
"""
        )

        ci_issue = (
            "Unused import `us:statutes/26/source#unused_rate`: imported symbol "
            "`unused_rate` is not referenced by any formula or proof import."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The copied rate uses the used rate.",
                skip_reviewers=True,
            )

        repaired_text = rulespec_file.read_text()
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert "unused_rate" not in repaired_text
        assert "used_rate" in repaired_text

    def test_generated_eval_repairs_positive_judgment_companions(self, tmp_path):
        repo = tmp_path / "rulespec-us-co"
        rulespec_file = repo / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
rules:
  - name: work_study_exemption
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: enrolled_in_work_study
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: existing_negative
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us-co:regulations/example#input.enrolled_in_work_study: false
  output:
    us-co:regulations/example#work_study_exemption: not_holds
"""
        )
        ci_issue = (
            "Judgment rule missing positive companion output coverage: "
            "`us-co:regulations/example#work_study_exemption` is not asserted "
            "as `holds` by the companion `.test.yaml` file."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures",
                return_value=[],
            ),
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Students in work study are exempt.",
                skip_reviewers=True,
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert any(
            case.get("output", {}).get("us-co:regulations/example#work_study_exemption")
            == "holds"
            for case in repaired_tests
        )

    def test_generated_eval_repairs_companions_with_unrelated_issues(self, tmp_path):
        repo = tmp_path / "rulespec-us-co"
        rulespec_file = repo / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
rules:
  - name: work_study_exemption
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: enrolled_in_work_study
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: existing_negative
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us-co:regulations/example#input.enrolled_in_work_study: false
  output:
    us-co:regulations/example#work_study_exemption: not_holds
"""
        )
        companion_issue = (
            "Judgment rule missing positive companion output coverage: "
            "`us-co:regulations/example#work_study_exemption` is not asserted "
            "as `holds` by the companion `.test.yaml` file."
        )
        unrelated_issue = (
            "Source scope mismatch: `work_study_exemption` is declared on "
            "`Person`, but the embedded source states a household/unit-scoped test."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult(
                        "ci",
                        passed=False,
                        issues=[companion_issue, unrelated_issue],
                    ),
                    ValidationResult("ci", passed=False, issues=[unrelated_issue]),
                ],
            ) as mock_ci,
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures",
                return_value=[],
            ),
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Students in work study are exempt.",
                skip_reviewers=True,
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        assert mock_ci.call_count == 2
        assert not metrics.ci_pass
        assert metrics.ci_issues == [unrelated_issue]
        assert any(
            case.get("output", {}).get("us-co:regulations/example#work_study_exemption")
            == "holds"
            for case in repaired_tests
        )

    def test_generated_eval_repairs_scalar_relation_rows(self, tmp_path):
        repo = tmp_path / "rulespec-us-co"
        rulespec_file = repo / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/7/2012/j#relation.member_of_household
rules:
  - name: household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, member_of_household.snap_member_is_elderly_or_disabled) > 0
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: elderly_case
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - true
  output:
    us-co:regulations/example#household_has_elderly_or_disabled_member: holds
"""
        )
        relation_issue = (
            "Test case `elderly_case` input invalid: relation "
            "`us:statutes/7/2012/j#relation.member_of_household` item #1 "
            "must be a mapping"
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[relation_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="An elderly or disabled member qualifies the household.",
                skip_reviewers=True,
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        rows = repaired_tests[0]["input"][
            "us:statutes/7/2012/j#relation.member_of_household"
        ]
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert rows == [
            {"us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled": True}
        ]

    def test_generated_eval_repairs_zero_branch_companions(self, tmp_path):
        repo = tmp_path / "rulespec-us-co"
        rulespec_file = repo / "regulations" / "example.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
rules:
  - name: benefit_limit
    kind: parameter
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: 100
  - name: benefit_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: 'if household_eligible: benefit_limit else: 0'
"""
        )
        rulespec_file.with_name("example.test.yaml").write_text(
            """- name: positive
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us-co:regulations/example#input.household_eligible: true
  output:
    us-co:regulations/example#benefit_amount: 100
"""
        )
        ci_issue = (
            "Zero branch test coverage missing: `benefit_amount` has a formula "
            "branch that returns 0, but no companion test asserts that output "
            "is zero."
        )
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", passed=True),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                side_effect=[
                    ValidationResult("ci", passed=False, issues=[ci_issue]),
                    ValidationResult("ci", passed=True, issues=[]),
                ],
            ) as mock_ci,
        ):
            metrics = _evaluate_generated_artifact_with_repairs(
                rulespec_file=rulespec_file,
                policy_repo_root=repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Eligible households receive a $100 benefit limit; otherwise zero.",
                skip_reviewers=True,
            )

        repaired_tests = yaml.safe_load(
            rulespec_file.with_name("example.test.yaml").read_text()
        )
        assert mock_ci.call_count == 2
        assert metrics.ci_pass
        assert any(
            case.get("output", {}).get("us-co:regulations/example#benefit_amount") == 0
            for case in repaired_tests
        )

    def test_test_input_assignment_ignores_formula_builtins(self):
        content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: deadline_days
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2025-01-01'
        formula: 7
  - name: deadline
    kind: derived
    entity: Household
    dtype: Date
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: date_add_days(application_date, deadline_days)
  - name: period_span_days
    kind: derived
    entity: Household
    dtype: Integer
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: days_between(period_start, period_end)
"""
        test_cases = [
            {
                "name": "deadline case",
                "input": {"#input.application_date": "2026-01-01"},
                "output": {
                    "#deadline": "2026-01-08",
                    "#period_span_days": 30,
                },
            }
        ]

        assert find_test_input_assignment_issues(content, test_cases) == []

    def test_numeric_occurrence_check_uses_embedded_operating_excerpt(self, tmp_path):
        rulespec_file = tmp_path / "statutes" / "7" / "2014" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: |-
    (a) Households in which each member receives qualifying public assistance shall be eligible.
rules:
  - name: snap_public_assistance_categorical_eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2008-10-01'
        formula: each_member_receives_qualifying_public_assistance
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "(a) Households in which each member receives qualifying public assistance shall be eligible.\n\n"
                    "(e) The unrelated standard deduction is 8.31 percent, $144, and $246."
                ),
            )

        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_inline_source_table_bounds(self, tmp_path):
        rulespec_file = tmp_path / "statutes" / "26" / "3241" / "b.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < 2.5: 1
          else: if average_account_benefits_ratio < 3.0: 2
          else: 3
  - name: section_3201_applicable_percentage_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.049
          2: 0.049
          3: 0
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "Tax rate schedule | Average account benefits ratio | Applicable percentage\n"
                    "| At least | But less than | Section 3201(b) |\n"
                    "| .............. | 2.5 | 4.9 |\n"
                    "| 2.5 | 3.0 | 4.9 |"
                ),
            )

        assert metrics.ci_pass
        assert metrics.missing_source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_skips_empty_deferred_artifact(self, tmp_path):
        rulespec_file = tmp_path / "statutes" / "wic" / "18901" / "5.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  status: deferred
  source_verification:
    corpus_citation_path: us-ca/statute/wic/18901.5
  summary: |-
    The department shall establish a program under 7 U.S.C. Sec. 2014(a). Categorical eligibility applies to households receiving or eligible to receive cash assistance under Part 5 (commencing with Section 17000), or food assistance under Chapter 10.1 (commencing with Section 18930).
  deferred_outputs:
    - output: us-ca:statutes/wic/18901/5#individual_categorically_eligible_for_calfresh
      reason: Requires upstream rules under Part 5 commencing with Section 17000 and Chapter 10.1 commencing with Section 18930, but no exact RuleSpec outputs were available in context.
rules: []
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "The department shall establish a program under 7 U.S.C. Sec. "
                    "2014(a). Categorical eligibility applies to households "
                    "receiving or eligible to receive cash assistance under Part "
                    "5 (commencing with Section 17000), or food assistance under "
                    "Chapter 10.1 (commencing with Section 18930)."
                ),
            )

        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_generated_numeric_grounding_uses_embedded_operating_excerpt(
        self, tmp_path
    ):
        rulespec_file = tmp_path / "statutes" / "7" / "2014" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: |-
    (a) Households in which each member receives qualifying public assistance shall be eligible.
rules:
  - name: unrelated_standard_deduction_amount
    kind: parameter
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: 144
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "(a) Households in which each member receives qualifying public assistance shall be eligible.\n\n"
                    "(e) The unrelated standard deduction is $144."
                ),
            )

        assert not metrics.ci_pass
        assert metrics.ungrounded_numeric_count == 1
        assert any("144" in issue for issue in metrics.ci_issues)

    def test_generated_numeric_grounding_uses_proof_excerpts_with_compact_summary(
        self, tmp_path
    ):
        rulespec_file = tmp_path / "statutes" / "26" / "3121" / "w.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3121
  summary: |-
    Church election timing rule.
rules:
  - name: election_timing_days_after_enactment_threshold
    kind: parameter
    dtype: Count
    period: Day
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: more than 90 days after July 18, 1984
    versions:
      - effective_from: '1990-01-01'
        formula: 90
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="A different paragraph contains $144.",
            )

        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.source_numeric_occurrence_count == 0

    def test_numeric_occurrence_check_counts_imported_named_scalars(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        child = policy_repo / "statutes" / "7" / "2015" / "d" / "2" / "B.yaml"
        child.parent.mkdir(parents=True)
        child.write_text(
            """format: rulespec/v1
rules:
  - name: dependent_child_age_exemption_threshold_years
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2008-10-01'
        formula: |-
          6
"""
        )
        parent = policy_repo / "statutes" / "7" / "2015" / "d" / "2.yaml"
        parent.write_text(
            """format: rulespec/v1
module:
  summary: |-
    A household member with responsibility for care of a dependent child under age 6 is exempt.
imports:
  - us:statutes/7/2015/d/2/B
rules:
  - name: person_exempt_from_paragraph_1_work_requirements
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2008-10-01'
        formula: care_responsibility_exemption_applies
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=parent,
                policy_repo_root=policy_repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="A household member with responsibility for care of a dependent child under age 6 is exempt.",
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_imported_numeric_concept_names(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us-co"
        federal_repo = tmp_path / "rulespec-us"
        child = (
            federal_repo
            / "policies"
            / "usda"
            / "snap"
            / "fy-2026-cola"
            / "income-eligibility-standards.yaml"
        )
        child.parent.mkdir(parents=True)
        child.write_text(
            """format: rulespec/v1
rules:
  - name: snap_gross_income_limit_130_percent_fpl_48_states_dc
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_gross_income_limit_130_percent_fpl_48_states_dc_table[household_size]
"""
        )
        parent = policy_repo / "regulations" / "example.yaml"
        parent.parent.mkdir(parents=True)
        parent.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Other households must meet the 130% gross income standard.
imports:
  - us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#snap_gross_income_limit_130_percent_fpl_48_states_dc
rules:
  - name: colorado_snap_gross_income_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_gross_income_limit_130_percent_fpl_48_states_dc
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=parent,
                policy_repo_root=policy_repo,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Other households must meet the 130% gross income standard.",
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_formula_identifier_numbers(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Children under age 18 qualify.
rules:
  - name: qualifying_child
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: child_under_18
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Children under age 18 qualify.",
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_verification_values(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    The standard deduction amounts are $209 for household sizes 1 through 3 and $223 for household size 4.
rules:
  - name: restates_standard_deduction
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction
      authority: federal
    verification:
      values:
        snap_standard_deduction_table:
          1: 209
          2: 209
          3: 209
          4: 223
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "The standard deduction amounts are $209 for household sizes "
                    "1 through 3 and $223 for household size 4."
                ),
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_counts_deferred_output_reasons(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  deferred_outputs:
    - output: example:provision#full_deduction_for_aged_households
      reason: This branch depends on whether the household contains a person aged 60 or older.
  summary: |-
    A deduction is 10 dollars unless the household has a person aged 60 or older.
rules:
  - name: base_deduction_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-01-01'
        formula: '10'
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "A deduction is 10 dollars unless the household has a "
                    "person aged 60 or older."
                ),
            )

        assert metrics.ci_pass
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_ignores_deferred_reason_section_numbers(
        self, tmp_path
    ):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  deferred_outputs:
    - output: example:provision#separate_branch
      reason: This branch is deferred until Section 4.000.1 is encoded.
rules: []
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The operative non-citation amount is 4 dollars.",
            )

        assert not metrics.ci_pass
        assert metrics.numeric_occurrence_issues == [
            "Source numeric value 4 appears 1 time(s), but only 0 named scalar "
            "definition(s) with that value were found."
        ]

    def test_repeated_source_scalar_is_covered_by_one_named_definition(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    2A. Where earnings are less than £20 in any week and would not exceed £20.
rules:
  - name: pc_special_employment_maximum_weekly_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2025-03-31'
        formula: 20
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "2A. Where earnings are less than £20 in any week and "
                    "would not exceed £20."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 2
        assert metrics.covered_source_numeric_occurrence_count == 2
        assert metrics.missing_source_numeric_occurrence_count == 0
        assert metrics.numeric_occurrence_issues == []

    def test_numeric_occurrence_check_ignores_section_cross_references(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: Households must receive an opportunity to participate within thirty days.
rules:
  - name: standard_opportunity_to_participate_deadline_days
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2025-01-01'
        formula: 30
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "Households shall receive an opportunity to participate within "
                    "thirty (30) calendar days. The office shall determine delay "
                    "cause as outlined in Sections 4.205.3 through 4.205.4."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.source_numeric_occurrence_count == 1
        assert metrics.numeric_occurrence_issues == []

    def test_ignores_bracketed_superseded_numeric_source_text(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: As of October 1, [2024] 2025, the allowance is [$31] $32.
rules:
  - name: telephone_standard_allowance_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 32
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "As of October 1, [2024] 2025, the allowance is [$31] $32."
                ),
            )

        assert metrics.ci_pass
        assert not any("31" in issue for issue in metrics.numeric_occurrence_issues)

    def test_preserves_bracketed_formula_numeric_source_text(self):
        numbers = validator_pipeline.extract_numbers_from_text(
            "684 to 4,999 | n = 150 + [ 0.1224(N-683)]"
        )

        assert 0.1224 in numbers
        assert 683 in numbers

    def test_accepts_pence_threshold_grounded_as_decimal_gbp(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    13. Small amounts of state pension credit

    Where the amount of state pension credit payable is less than 10 pence per week,
    the credit shall not be payable unless the claimant is in receipt of another benefit
    payable with the credit.
rules:
  - name: small_amount_threshold
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2025-03-21'
        formula: 0.10
  - name: amount_payable
    kind: input
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
  - name: is_payable
    kind: derived
    entity: Person
    dtype: Boolean
    period: Week
    versions:
      - effective_from: '2025-03-21'
        formula: amount_payable >= small_amount_threshold
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text=(
                    "Where the amount of state pension credit payable is less than "
                    "10 pence per week, the credit shall not be payable."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.missing_source_numeric_occurrence_count == 0

    def test_runs_generalist_reviewer_and_records_result(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: Provision text with £10.
rules:
  - name: example_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: 10
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])
        reviewer_result = ValidationResult(
            "generalist-reviewer",
            False,
            score=4.5,
            issues=["Merged distinct statutory branches."],
        )

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
            patch.object(
                ValidatorPipeline, "_run_reviewer", return_value=reviewer_result
            ) as mock_reviewer,
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="Provision text with £10.",
            )

        assert metrics.compile_pass is True
        assert metrics.ci_pass is True
        assert metrics.generalist_review_pass is False
        assert metrics.generalist_review_score == 4.5
        assert metrics.generalist_review_issues == [
            "Merged distinct statutory branches."
        ]
        mock_reviewer.assert_called_once()
        assert mock_reviewer.call_args.args[0] == "generalist-reviewer"
        assert "atomic source slice" in mock_reviewer.call_args.kwargs["review_context"]
        assert (
            "stale, generic, or misleading"
            in mock_reviewer.call_args.kwargs["review_context"]
        )

    def test_timing_clause_review_context_mentions_boolean_day_predicate(
        self, tmp_path
    ):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: On the first day of the next benefit week.
rules:
  - name: example_timing_rule
    kind: parameter
    entity: Person
    dtype: Boolean
    period: Day
    versions:
      - effective_from: '2025-01-01'
        formula: true
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])
        reviewer_result = ValidationResult(
            "generalist-reviewer",
            True,
            score=8.0,
            issues=[],
        )

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
            patch.object(
                ValidatorPipeline, "_run_reviewer", return_value=reviewer_result
            ) as mock_reviewer,
        ):
            evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="On the first day of the next benefit week.",
            )

        assert (
            "boolean day-predicate helper"
            in mock_reviewer.call_args.kwargs["review_context"]
        )

    def test_build_eval_prompt_for_uk_timing_leaf_discourages_invented_day_offsets(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the assessed amount comprises income from capital, it shall be "
                "deemed to increase or decrease on the first day of the next benefit "
                "week to commence on or after the day on which the income increases "
                "or decreases."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Do not convert relative temporal phrases" in prompt
        assert "`*_offset = 1`" in prompt

    def test_build_eval_prompt_for_atomic_conjunctive_branch_discourages_normative_names(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the Secretary of State is informed that the arrangements under "
                "which the assessed amount is paid contains provision—\n\n"
                "(b)\n\n"
                "for the date on which the increase is to be paid; and"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "atomic conjunctive branch slices" in prompt
        assert "do not pretend to encode the whole parent consequence" in prompt
        assert "avoid standalone normative names like `..._must_...`" in prompt
        assert "do not make the principal output a bare input stub" in prompt
        assert "feed the asserted output back into `input:`" in prompt
        assert "treat the carve-out as displacing this slice" in prompt

    def test_build_eval_prompt_for_comparative_month_apart_phrase_discourages_numeric_thresholds(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the last four payments if the last two payments are less than one month apart; or"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "less than one month apart" in prompt
        assert "one_month_threshold = 1" in prompt
        assert "the `one month` comparator is not a standalone numeric scalar" in prompt
        assert "do not invent `1`-valued threshold/count helpers" in prompt
        assert (
            "branch-specific output is a `Count` or other non-Boolean basis selector"
            in prompt
        )
        assert "do not write an inline conditional without `else`" in prompt
        assert (
            "negative tests should usually assert only the `_applies` boolean" in prompt
        )
        assert (
            "expect the principal basis-count output to remain the active legal basis"
            in prompt
        )
        assert "trigger decomposed-date CI failures" in prompt

    def test_build_eval_prompt_for_single_payment_period_discourages_parallel_units(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "where the period in respect of which a payment is made exceeds a week, "
                "and in a case where that period is three months, the amount is calculated ..."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "keep one canonical fact or classification for that single period" in prompt
        )
        assert "parallel free inputs like `*_in_weeks` and `*_in_months`" in prompt
        assert "do not require a second independent duration input" in prompt
        assert (
            "do not feed the same legal period through contradictory units or categories"
            in prompt
        )

    def test_build_eval_prompt_for_amount_included_determination_requires_applicability_bound_money_output(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the amount to be included in the claimant's weekly income shall be determined—\n\n"
                "(ii)\n\n"
                "in a case where that period is three months, by multiplying the amount of the payment by 4 and dividing the product by 52;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not leave that money output unconditional" in prompt
        assert "typically with an explicit `else: 0`" in prompt
        assert (
            "paragraph-level exceptions or a different payment period displace the limb"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_includes_leaf_discourages_blanket_negation(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                'Subject to paragraphs (3), (4) and (4A), "earnings" in the case '
                "of employment as an employed earner, means any remuneration or "
                "profit derived from that employment and includes—\n\n"
                "(a)\n\n"
                "any bonus or commission;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Subject to paragraphs (3), (4) and (4A), ... includes—" in prompt
        assert "blanket negating gate" in prompt
        assert "Do not make a composite `subject_to_*_satisfied`" in prompt
        assert "branch-specific fact gate" in prompt
        assert "permits this branch to count" in prompt
        assert (
            "do not collapse all cited qualifications into one opaque helper" in prompt
        )
        assert (
            "one paragraph-specific qualification input or import per cited paragraph"
            in prompt
        )

    def test_build_eval_prompt_for_payment_level_slice_discourages_blind_unsupported_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Except where paragraph (2) and (4) apply, where the period in respect "
                "of which a payment is made does not exceed a week, the whole of that "
                "payment shall be included in the claimant's weekly income."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "individual payment or `that payment`" in prompt
        assert "preserve that payment-scoped subject" in prompt
        assert "prefer `entity: Payment`" in prompt
        assert "prefer `entity: Asset`" in prompt
        assert "provide per-payment rows under `tables:`" in prompt
        assert "provide per-item rows under `tables:`" in prompt
        assert "exact entity name `Payment:`" in prompt
        assert "Use `status: entity_not_supported`" in prompt
        assert "only as a last resort" in prompt
        assert "Do not prefer that fallback" in prompt

    def test_build_eval_prompt_for_except_where_and_citations_discourages_joint_exception(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Except where paragraph (2) and (4) apply, the amount to be included "
                "shall be determined—"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Except where paragraph (2) and (4) apply" in prompt
        assert (
            "do not assume the exception is displaced only when both cited paragraphs apply simultaneously"
            in prompt
        )
        assert (
            "treat the slice as inoperative when any cited paragraph applies" in prompt
        )

    def test_build_eval_prompt_for_payable_phrase_preserves_payability_fact(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "statutory sick pay and statutory maternity pay payable by the "
                "employer under the 1992 Act;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "model payability as the legal fact" in prompt
        assert "Do not replace `payable` with `receives` or `received`" in prompt

    def test_build_eval_prompt_for_regular_pattern_clause_preserves_full_qualifier(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the claimant's regular pattern of work is such that he does not "
                "work the same hours every week;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "regular pattern of work is such that" in prompt
        assert (
            "Do not shorten the branch to only `does not work the same hours every week`"
            in prompt
        )

    def test_build_eval_prompt_for_enumerated_payments_discourages_or_collapse(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "statutory sick pay and statutory maternity pay payable by the "
                "employer under the 1992 Act;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not collapse them into a single `x_or_y` principal output" in prompt
        assert "statutory_sick_pay_or_statutory_maternity_pay_*" in prompt

    def test_build_eval_prompt_for_branch_slice_preserves_binding_lead_in_conjuncts(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "This paragraph applies where the period which—\n\n"
                "is a period of the same length as the period in respect of which "
                "the last payment of the pre-increase assessed amount was made.\n\n"
                "(b)\n\n"
                "ends on the first increased payment date,"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "distinguish mere placement context from binding lead-in conjuncts"
            in prompt
        )
        assert "preserve both conjuncts" in prompt
        assert "do not drop the same-length requirement" in prompt

    def test_build_eval_prompt_for_where_on_branch_discourages_material_implication(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13B",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "All benefits except those mentioned in paragraph (1) shall be treated as paid—\n\n"
                "(b)\n\n"
                "where the benefit is paid in arrears, on the last day of the benefit week "
                "in which the benefit is payable."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13B",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13B.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "treat `X` and `Y` as a positive conjunction for this branch" in prompt
        assert "Do not rewrite that as material implication like `not X or Y`" in prompt
        assert (
            "if the branch-triggering condition itself is false, the branch-specific output should usually be `false`"
            in prompt
        )

    def test_build_eval_prompt_for_disjunctive_payment_description_preserves_qualifier_scope(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/15",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of section 15(1)(j) (income to include income of prescribed descriptions), "
                "income of the following descriptions is prescribed—\n\n"
                "(ac)\n\n"
                "any retired pay, pension or allowance granted in respect of disablement or any pension or "
                "allowance granted to a widow, widower or surviving civil partner in respect of a death due to "
                "service or war injury under an instrument specified in section 639(2) of the Income Tax "
                "(Earnings and Pensions) Act 2003, where such payment does not fall within paragraph (a) of the "
                "definition of “war disablement pension” in section 17(1) of the State Pension Credit Act 2002 or, "
                "in respect of any retired pay or pension granted in respect of disablement, where such payment "
                "does not fall within paragraph (b) of that definition;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the scope of the first qualifier across every antecedent payment type it grammatically modifies"
            in prompt
        )
        assert (
            "do not narrow the first `where ...` clause to only the later-mentioned category"
            in prompt
        )
        assert "preserve the paragraph-(a) path for retired pay and pension" in prompt

    def test_build_eval_prompt_for_royalties_slice_preserves_consideration_scope(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/5/a",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "PART III Income\n\n"
                "17. Calculation of weekly income\n\n"
                "(5)\n\n"
                "This paragraph applies to—\n\n"
                "(a)\n\n"
                "royalties or other sums received as a consideration for the use of, or the "
                "right to use, any copyright, design, patent or trade mark;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-5-a.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the consideration-for-use/right-to-use qualifier across both `royalties` and `other sums`"
            in prompt
        )
        assert "do not model `royalty` as a free-standing qualifying limb" in prompt
        assert "a bare `payment_is_royalty` fact is too broad" in prompt

    def test_build_eval_prompt_for_employed_earner_definition_preserves_shared_qualifier(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/5",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "17A. Earnings of an employed earner\n\n"
                "(5)\n\n"
                "In this regulation “employed earner” means a person who is gainfully "
                "employed in Great Britain either under a contract of service, or in an "
                "office (including elective office) with emoluments chargeable to income "
                "tax under Schedule E."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/5",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-5.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the shared qualifying employment/office across each alternative limb"
            in prompt
        )
        assert (
            "do not decompose the rule into one free-standing `person_is_X` fact plus separate `under_A` and `in_B` facts"
            in prompt
        )
        assert (
            "distribute the shared qualifier across the alternatives with branch-specific combined facts"
            in prompt
        )

    def test_build_eval_prompt_for_complete_capital_bands_discourages_fractional_division(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/15",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of section 15(2) (deemed income from capital) and subject to "
                "regulation 17(8) (capital to be disregarded), a claimant’s capital shall be "
                "deemed to yield a weekly income of—\n\n"
                "(a)\n\n"
                "£1 for each £500 in excess of £10,000; and"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "treat `for each £500` as counting complete bands, not proportional fractions"
            in prompt
        )
        assert "derive the band count with `floor(excess / band_size)`" in prompt
        assert (
            "include a non-exact-multiple excess case like `£750` above threshold"
            in prompt
        )


class TestGeneratedBundleCleaning:
    def test_clean_generated_file_content_strips_fence_and_trailing_prose(self):
        content = (
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "The encoding captures the enhanced rate."
        )

        cleaned = _clean_generated_file_content(content)

        assert cleaned == (
            "- name: base\n  output:\n    child_benefit_enhanced_rate: 26.05\n"
        )

    def test_clean_generated_file_content_strips_inline_currency_suffixes(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The enhanced rate is 26.05 GBP.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05 GBP\n"
        )

        cleaned = _clean_generated_file_content(content)

        assert "26.05 GBP" not in cleaned.split("formula:", 1)[1]
        assert "formula: 26.05" in cleaned

    def test_clean_generated_file_content_repairs_yaml_apostrophe_escapes(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            '  summary: "Double quoted taxpayer\\\'s amount"\n'
            "rules:\n"
            "  - name: military_retirement_benefits_definition\n"
            "    kind: derived\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            "              excerpt: 'benefits received as a result of the individual\\'s service'\n"
            "    versions:\n"
            "      - effective_from: '2019-01-01'\n"
            "        formula: benefit_amount\n"
        )

        cleaned = _clean_generated_file_content(content)
        payload = yaml.safe_load(cleaned)

        assert payload["module"]["summary"] == "Double quoted taxpayer's amount"
        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == "benefits received as a result of the individual's service"

    def test_clean_generated_file_content_repairs_semicolon_excerpts(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Payment exclusions.\n"
            "rules:\n"
            "  - name: excluded_payment_limit\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            '              excerpt: "per capita Payments ... of two thousand dollars ($2,000) or less"; "up to two thousand dollars ($2,000) per year"; "The first two thousand dollars ($2,000) of each payment is excluded"\n'
            "    versions:\n"
            "      - effective_from: '0001-01-01'\n"
            "        formula: 2000\n"
        )

        cleaned = _clean_generated_file_content(content)
        payload = yaml.safe_load(cleaned)

        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == (
            "per capita Payments ... of two thousand dollars ($2,000) or less; "
            "up to two thousand dollars ($2,000) per year; "
            "The first two thousand dollars ($2,000) of each payment is excluded"
        )

    def test_clean_generated_file_content_repairs_conjoined_quoted_excerpts(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Fuel cell cap.\n"
            "rules:\n"
            "  - name: residential_clean_energy_fuel_cell_credit_component\n"
            "    kind: derived\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            '              excerpt: "applicable percentages of qualified fuel cell property expenditures" and "shall not exceed $500 with respect to each half kilowatt of capacity"\n'
            "    versions:\n"
            "      - effective_from: '2006-01-01'\n"
            "        formula: 500\n"
        )

        cleaned = _clean_generated_file_content(content)
        payload = yaml.safe_load(cleaned)

        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == (
            "applicable percentages of qualified fuel cell property expenditures "
            "and shall not exceed $500 with respect to each half kilowatt of capacity"
        )

    def test_materialize_eval_artifact_rejects_non_rulespec_bundle(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = (
            "=== FILE: example.yaml ===\nrules:\n  - name: missing_format_header\n"
        )

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_eval_artifact_repairs_single_file_conjoined_excerpts(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "section-25d.yaml"
        response = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: Fuel cell cap.\n"
            "rules:\n"
            "  - name: residential_clean_energy_fuel_cell_credit_component\n"
            "    kind: derived\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - source:\n"
            '              excerpt: "applicable percentages of qualified fuel cell property expenditures" and "shall not exceed $500 with respect to each half kilowatt of capacity"\n'
            "    versions:\n"
            "      - effective_from: '2006-01-01'\n"
            "        formula: |-\n"
            "          if expenditures_after_termination_date:\n"
            "              0\n"
            "          else:\n"
            "              base_expenditures =\n"
            "                  max(0, qualified_solar_electric_property_expenditures)\n"
            "                  + max(0, qualified_battery_storage_technology_expenditures)\n"
            "              credit = residential_clean_energy_applicable_percentage * base_expenditures\n"
            "              max(0, credit + residential_clean_energy_fuel_cell_credit_component)\n"
        )

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        payload = yaml.safe_load(output_file.read_text())
        excerpt = payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "excerpt"
        ]
        assert excerpt == (
            "applicable percentages of qualified fuel cell property expenditures "
            "and shall not exceed $500 with respect to each half kilowatt of capacity"
        )
        formula = payload["rules"][0]["versions"][0]["formula"]
        assert "base_expenditures =" not in formula
        assert "credit =" not in formula
        assert (
            "residential_clean_energy_applicable_percentage * "
            "(max(0, qualified_solar_electric_property_expenditures)"
        ) in formula

    def test_materialize_eval_artifact_cleans_bundled_rulespec_fences(self, tmp_path):
        output_file = tmp_path / "source" / "uksi-2006-965-regulation-2.yaml"
        llm_response = (
            "=== FILE: uksi-2006-965-regulation-2.yaml ===\n"
            "```yaml\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The enhanced rate is £26.05.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05\n"
            "```\n"
            "=== FILE: uksi-2006-965-regulation-2.test.yaml ===\n"
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "Trailing prose.\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file)

        assert wrote is True
        assert output_file.read_text().startswith("format: rulespec/v1\n")
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert "child_benefit_enhanced_rate: 26.05" in test_text
        assert "period: '2025-04-07'" in test_text

    def test_materialize_eval_artifact_salvages_rulespec_workspace_files_when_response_is_summary(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-2025-03-31.yaml"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The weekly amount is £19.30.\n"
            "rules:\n"
            "  - name: pc_housing_non_dependant_deduction_other_weekly_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-03-21'\n"
            "        formula: 19.30\n"
        )
        (workspace_root / output_file.with_suffix(".test.yaml").name).write_text(
            "- name: base_case\n"
            "  period: 2025-04-01\n"
            "  input: {}\n"
            "  output:\n"
            "    pc_housing_non_dependant_deduction_other_weekly_amount: 19.30\n"
        )

        wrote = _materialize_eval_artifact(
            "Both files written.",
            output_file,
            source_text="£19.30",
            workspace_root=workspace_root,
        )

        assert wrote is True
        assert output_file.read_text().startswith("format: rulespec/v1\n")
        assert output_file.with_suffix(".test.yaml").exists()

    def test_materialize_eval_artifact_rejects_non_rulespec_workspace_main(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "example.yaml"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            "rules:\n  - name: missing_format_header\n"
        )

        wrote = _materialize_eval_artifact(
            "Both files written.",
            output_file,
            workspace_root=workspace_root,
        )

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_eval_artifact_normalizes_rulespec_test_periods(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451, effective October 1, 2025.
rules:
  - name: snap_standard_utility_allowance
    kind: parameter
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 451
=== FILE: example.test.yaml ===
- name: pre_effective_zero
  period: 2025-09
  input: {}
  output:
    snap_standard_utility_allowance: 0
- name: applies
  period: 2026-01
  input: {}
  output:
    snap_standard_utility_allowance: 451
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert "pre_effective_zero" not in test_text
        assert "period: 2026-01" in test_text

    def test_materialize_eval_artifact_normalizes_quoted_date_outputs(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: The deadline is seven days after application.
rules:
  - name: deadline_days
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2025-01-01'
        formula: 7
  - name: deadline
    kind: derived
    entity: Household
    dtype: Date
    period: Month
    versions:
      - effective_from: '2025-01-01'
        formula: date_add_days(application_date, deadline_days)
=== FILE: example.test.yaml ===
- name: applies
  period: 2026-01
  input:
    '#input.application_date': '2026-01-01'
  output:
    '#deadline': '2026-01-08'
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        payload = yaml.safe_load(test_text)
        assert payload[0]["input"]["#input.application_date"] == "2026-01-01"
        assert payload[0]["output"]["#deadline"] == date(2026, 1, 8)

    def test_normalize_test_periods_repairs_misindented_period_end(self):
        rulespec_text = """format: rulespec/v1
module:
  summary: Section defines an annual table.
rules:
  - name: annual_rate
    kind: parameter
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 0.05
"""
        test_text = _normalize_test_periods_to_effective_dates(
            """- name: ratio_at_9_0
  period:
    period_kind: tax_year
    start: '2026-01-01'
  end: '2026-12-31'
  input: {}
  output:
    annual_rate: 0.05
""",
            rulespec_content=rulespec_text,
        )

        cases = yaml.safe_load(test_text)

        assert cases[0]["period"] == {
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        }
        assert "end" not in cases[0]

    def test_normalize_test_periods_rewrites_iso_week_shorthand(self):
        rulespec_text = """format: rulespec/v1
module:
  summary: Weekly child benefit rate.
rules:
  - name: child_benefit_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '0001-01-01'
        formula: 27.05
"""
        test_text = _normalize_test_periods_to_effective_dates(
            """- name: weekly_case
  period: 2025-W01
  input: {}
  output:
    child_benefit_weekly_rate: 27.05
""",
            rulespec_content=rulespec_text,
        )

        cases = yaml.safe_load(test_text)

        assert cases[0]["period"] == {
            "period_kind": "benefit_week",
            "start": "2024-12-30",
            "end": "2025-01-05",
        }

    def test_materialize_eval_artifact_normalizes_mapping_style_tests_to_list(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-regulation-10-5-b-ii.yaml"
        response = """=== FILE: uksi-2002-1792-regulation-10-5-b-ii.yaml ===
format: rulespec/v1
module:
  summary: The day referred to in branch ii is true when the condition applies.
rules:
  - name: day_referred_to_10_5_b_ii
    kind: derived
    entity: Person
    dtype: Boolean
    period: Day
    versions:
      - effective_from: '2025-03-21'
        formula: some_fact
=== FILE: uksi-2002-1792-regulation-10-5-b-ii.test.yaml ===
case_branch_ii_applies:
  period: 2025-03-21
  input:
    some_fact: true
  output:
    day_referred_to_10_5_b_ii: true
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert test_text.lstrip().startswith("- ")
        assert "name: case_branch_ii_applies" in test_text
        assert "case_branch_ii_applies:" not in test_text

    def test_normalize_test_periods_drops_speculative_pre_effective_zero_case_for_monthly_update(
        self,
    ):
        rulespec_text = """format: rulespec/v1
module:
  summary: The SUA is $451 effective October 1, 2025.
rules:
  - name: snap_standard_utility_allowance
    kind: parameter
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 451
"""
        source_text = (
            "Current-effective Tennessee utility allowance slice.\n\n"
            "The Standard Utility Allowance (SUA) is used when the household is\n"
            "responsible for heating or cooling costs.\n"
            "The SUA is $451, effective October 1, 2025.\n"
        )
        test_text = _normalize_test_periods_to_effective_dates(
            "- name: applies\n"
            "  period: 2026-01\n"
            "  output:\n"
            "    snap_standard_utility_allowance: 451\n"
            "- name: pre_effective_month_zero\n"
            "  period: 2025-09\n"
            "  output:\n"
            "    snap_standard_utility_allowance: 0\n",
            rulespec_content=rulespec_text,
            source_text=source_text,
        )

        assert "pre_effective_month_zero" not in test_text
        assert "period: 2026-01" in test_text

    def test_normalize_test_case_value_preserves_invalid_numeric_expression(self):
        assert _normalize_test_case_value("30 / 0") == "30 / 0"

    def test_materialize_eval_artifact_adds_missing_oracle_hint_output_from_rulespec(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: Homeless Shelter Deduction - $198.99.
rules:
  - name: snap_homeless_shelter_deduction_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 198.99
  - name: snap_homeless_shelter_deduction_available
    kind: parameter
    dtype: Boolean
    versions:
      - effective_from: '2025-10-01'
        formula: true
=== FILE: example.test.yaml ===
- name: base
  period: 2025-10
  output:
    snap_homeless_shelter_deduction_amount: 198.99
"""

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            policyengine_rule_hint="snap_homeless_shelter_deduction_available",
        )

        assert wrote is True
        test_payload = yaml.safe_load(output_file.with_suffix(".test.yaml").read_text())
        assert (
            test_payload[0]["output"]["snap_homeless_shelter_deduction_available"]
            is True
        )

    def test_materialize_eval_artifact_uses_canonical_oracle_hint_output_key(
        self, tmp_path
    ):
        output_file = (
            tmp_path / "rulespec-us" / "policies" / "usda" / "snap" / "homeless.yaml"
        )
        response = """=== FILE: homeless.yaml ===
format: rulespec/v1
module:
  summary: Homeless Shelter Deduction availability.
rules:
  - name: snap_homeless_shelter_deduction_available
    kind: parameter
    dtype: Boolean
    versions:
      - effective_from: '2025-10-01'
        formula: true
=== FILE: homeless.test.yaml ===
- name: base
  period: 2025-10
  output:
    snap_homeless_shelter_deduction_available: true
"""

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            policyengine_rule_hint="snap_homeless_shelter_deduction_available",
        )

        assert wrote is True
        output = yaml.safe_load(output_file.with_suffix(".test.yaml").read_text())[0][
            "output"
        ]
        assert "snap_homeless_shelter_deduction_available" not in output
        assert (
            output[
                "us:policies/usda/snap/homeless#snap_homeless_shelter_deduction_available"
            ]
            is True
        )

    def test_can_include_policyengine_metrics_for_uk_artifact(self, tmp_path):
        rules_file = tmp_path / "source" / "uksi-2006-965-regulation-2.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: https://www.legislation.gov.uk/uksi/2006/965/regulation/2 states the enhanced rate is £26.05.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)
        pe_result = ValidationResult(
            "policyengine",
            passed=True,
            score=1.0,
            issues=[],
        )

        with (
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
                return_value=compile_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_ci",
                return_value=ci_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_policyengine",
                return_value=pe_result,
            ) as mock_policyengine,
        ):
            metrics = evaluate_artifact(
                rulespec_file=rules_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules-engine"),
                source_text="The enhanced rate is £26.05 from 2025-04-07.",
                oracle="policyengine",
                policyengine_country="uk",
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.policyengine_pass is True
        assert metrics.policyengine_score == 1.0
        assert metrics.policyengine_issues == []
        mock_policyengine.assert_called_once()


class TestEvalPrompt:
    def test_build_eval_prompt_includes_rulespec_schema_guardrails(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Grant standard is 165 for one child.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1.yaml",
            include_tests=True,
        )

        assert (
            "Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`."
            in prompt
        )
        assert "entity:" in prompt
        assert "period:" in prompt
        assert "dtype:" in prompt
        assert "RuleSpec requirements:" in prompt
        assert "The RuleSpec file must begin with `format: rulespec/v1`" in prompt
        assert (
            "Use chained `if condition: value else: other_value` expressions" in prompt
        )
        assert "Do not write `else if` or `elif`" in prompt
        assert "do not inline that cross-reference's mechanics into this file" in prompt
        assert (
            "additional_standard_deduction_entitlement_count_under_subsection_f"
            in prompt
        )
        assert "Do not start a local input with" in prompt
        assert "_under_section_<section>" in prompt
        assert "completed federal\n  return amount" in prompt
        assert "itemized_deductions_claimed_on_federal_return" in prompt
        assert "For IRC section 22" not in prompt
        assert "dependent_of_tax_unit" in prompt
        assert "only the exception input changes" in prompt
        assert (
            "Do not replace a specific upstream output with a broad local input"
            in prompt
        )
        assert "only one entity type" in prompt
        assert "Do not assert relation-child outputs" in prompt
        assert "Do not use bare year periods like `2024`" in prompt
        assert "Never encode US tax filing status" in prompt
        assert "Do not create local `#input.filing_status` facts" in prompt
        assert 'If the source says only "joint return"' in prompt
        assert 'status 4 falls under any "other case" branch' in prompt
        assert "Hard requirement for IRC section 151(d)" not in prompt
        assert "must use the numeric `filing_status` enum input directly" not in prompt
        assert "Importing an adjacent upstream output only as proof" in prompt
        assert "does not satisfy the dependency" in prompt
        assert "is not an executable dependency" in prompt
        assert "Never drop the jurisdiction prefix" in prompt
        assert "listed under invalid copied local inputs" in prompt
        assert "do not preserve, rename, or recreate" in prompt
        assert "file-level import without a `#symbol` fragment" in prompt
        assert "except for purposes of subsection (a)" in prompt
        assert "Do not add a fragment import only for proof" in prompt
        assert "treated as attributable to" in prompt
        assert "amount-level" in prompt
        assert "boolean or `dtype: Judgment` predicate" in prompt

    def test_build_eval_prompt_for_broad_application_clause_discourages_passthrough_outputs(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="7 USC 2014(a)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Participation in the supplemental nutrition assistance program "
                "shall be limited to those households whose incomes and other "
                "financial resources are determined to be a substantial limiting "
                "factor in permitting them to obtain a more nutritious diet. "
                "Assistance under this program shall be furnished to all eligible "
                "households who make application for such participation."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "7 USC 2014(a)",
            "cold",
            workspace,
            [],
            target_file_name="statutes/7/2014/a.yaml",
            target_ref_prefix="us:statutes/7/2014/a",
            include_tests=True,
        )

        assert (
            "broad application, furnishing, administrative duty, or purpose clause"
            in prompt
        )
        assert (
            "do not create an executable derived output just to paraphrase it" in prompt
        )
        assert "assistance shall be furnished to all eligible households" in prompt
        assert (
            "Do not encode a pure pass-through rule whose formula is only one local fact"
            in prompt
        )
        assert "one-time" in prompt
        assert "more than one consecutive month" in prompt
        assert "Do not append citation or file suffixes like `_2014_a`" in prompt
        assert (
            "For every encoded `except`, `unless`, or `notwithstanding` carve-out"
            in prompt
        )
        assert 'shall not apply" or "does not apply"' in prompt
        assert "that helper as `holds`" in prompt
        assert '"through such account"' in prompt
        assert "sets that exception input true" in prompt
        assert "Do not collapse a list of cited exceptions" in prompt
        assert "Do not create derived `dtype: Boolean` helper rules" in prompt

    def test_build_eval_prompt_includes_supported_schema_enums(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 25.60 ... 16.95",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "Do not invent arbitrary entities." in prompt
        assert (
            "Standard `entity:` examples are `Payment`, `Person`, `TaxUnit`, `Household`, "
            "`Family`, `TanfUnit`, `SnapUnit`, `SPMUnit`, `Corporation`, `Business`, "
            "`Employer`, `Asset`, `StateAgency`."
        ) in prompt
        assert "introduce a narrow singular" in prompt
        assert "Allowed `period:` values are `Year`, `Month`, `Week`, `Day`." in prompt
        assert "do not use ISO week shorthands like `2025-W01`" in prompt
        assert (
            "Allowed `dtype:` values are `Money`, `Rate`, `Boolean`, `Integer`, "
            "`Count`, `String`, `Decimal`, `Float`, or `Enum[Name]`."
        ) in prompt

    def test_build_eval_prompt_includes_unsupported_entity_fallback(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="ukpga/2010/1/section/1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) cease to be in force",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "ukpga/2010/1/section/1",
            "cold",
            workspace,
            [],
            target_file_name="ukpga-2010-1-section-1.yaml",
            include_tests=True,
        )

        assert (
            "If the source cannot be represented faithfully with the supported schema"
            in prompt
        )
        assert "`module.status: entity_not_supported`" in prompt
        assert "`module.status: deferred`" in prompt
        assert (
            "In a mixed provision, omit or defer only the affected executable" in prompt
        )
        assert "module.deferred_outputs[]" in prompt
        assert "do not model that numeric term as a local" in prompt
        assert "output` target path must include that source path segment" in prompt
        assert "Do not create tests for deferred" in prompt
        assert (
            "only when no executable rule in the requested source can be represented"
            in prompt
        )
        assert "leave the companion `.test.yaml` empty" in prompt
        assert "assertions against deferred symbols" in prompt

    def test_build_eval_prompt_for_filing_status_upstream_sources_requires_executable(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 7703",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "(a) General rule The determination of whether an individual is "
                "married shall be made as of the close of his taxable year."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 7703",
            "cold",
            workspace,
            [],
            target_file_name="7703.yaml",
            include_tests=True,
        )

        assert "Hard requirement for IRC sections 2, 6013, and 7703" not in prompt
        assert "section 151 deduction is `allowed` or `allowable`" not in prompt
        assert "Never introduce an import cycle" in prompt
        assert "same rule's name" in prompt
        assert "rate-bearing source" in prompt
        assert "cycle with a foundational base definition" in prompt
        assert "rate or rate" in prompt
        assert "source-named numeric boundary input" in prompt

    def test_build_eval_prompt_for_editorially_omitted_slice_allows_deferred_docstring(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/10/c",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "(c)\n\n"
                ". . . . . . . . . . . . . . . . . . . . . . . ."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/10/c",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-10-c.yaml",
            include_tests=True,
        )

        assert (
            "editorially omitted or repealed text shown by ellipses or dotted placeholders"
            in prompt
        )
        assert "leave `.test.yaml` empty" in prompt

    def test_build_eval_prompt_forbids_python_inline_ternaries(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "Do not use Python inline ternaries" in prompt
        assert "`x if cond else y`" in prompt

    def test_build_eval_prompt_requires_rulespec_conditional_expression_syntax(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "`if condition: value else: other_value`" in prompt
        assert "do not use YAML-style `if:` / `then:` / `else:` blocks" in prompt
        assert (
            "Do not append a multiline conditional directly onto another expression"
            in prompt
        )

    def test_build_eval_prompt_requires_decimal_ratios_for_rate_dtype(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The percentage prescribed is 60 per cent.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-7.yaml",
            include_tests=True,
        )

        assert (
            "For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals"
            in prompt
        )
        assert "never as arithmetic like `25 / 100`" in prompt
        assert "source.corpus_citation_path` is sufficient" in prompt
        assert (
            "Do not respond with summaries, markdown prose, or file-write confirmations"
            in prompt
        )
        assert "concise exact audit excerpt" in prompt
        assert "not the full source text" in prompt
        assert (
            "do not use inline assignment syntax like `:=` inside formula blocks"
            in prompt
        )

    def test_build_eval_prompt_for_uk_leaf_prefers_person_over_family_constant(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-07.\n\n"
                "The weekly rate of child benefit payable in respect of a child "
                "or qualifying young person shall be 26.05."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert (
            'Prefer `Person` when the source states an amount or condition "in respect of"'
            in prompt
        )
        assert (
            "do not collapse it into an unconditional family-level constant" in prompt
        )

    def test_build_eval_prompt_for_uk_pence_threshold_requires_gbp_decimal_and_weekly_cadence(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "Where the amount of state pension credit payable is less than 10 pence per week, "
                "the credit shall not be payable unless the claimant is in receipt of another "
                "benefit payable with the credit."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "include `unit: GBP`" in prompt
        assert "`10 pence` should become `0.10`, not `10`" in prompt
        assert "do not disguise it as arithmetic like `1 / 10`" in prompt
        assert "prefer a money variable with matching `period:` cadence" in prompt

    def test_build_eval_prompt_for_positive_conditional_uk_leaf_requires_zero_or_false_else_case(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/paragraph/4/1/a/iva",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "£20 is disregarded if the claimant or, if he has a partner, his partner "
                "is in receipt of Scottish adult disability living allowance."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/paragraph/4/1/a/iva",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-schedule-vi-paragraph-4-1-a-iva.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "positive conditional leaves" in prompt
        assert (
            "the inapplicable case should usually be `0` for `dtype: Money` or `false` for `dtype: Boolean`"
            in prompt
        )
        assert "do not use an unconditional amount or `else: true`" in prompt
        assert (
            "fixed supplement, allowance, or addition is payable only while an eligibility condition holds"
            in prompt
        )
        assert "do not leave that money output unconditional" in prompt
        assert (
            "do not collapse source-stated component facts into an opaque local input like `*_eligible_for_*`"
            in prompt
        )
        assert "prefer direct facts like `client_is_pregnant_parent`" in prompt

    def test_build_eval_prompt_for_determination_limb_discourages_invented_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the weekly amount of that claimant's income shall be determined—\n\n"
                "(i)\n\n"
                "if there is a recognised cycle of work, by reference to his average "
                "weekly income over the period of the complete cycle; or"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "do not invent sibling outcomes for non-applicable cases with `else: 0`"
            in prompt
        )
        assert "leave other cases to sibling limbs" in prompt
        assert "keep a branch-specific money or rate output for that basis" in prompt
        assert (
            "do not invent sibling outcomes for inapplicable cases with `else: 0`"
            in prompt
        )
        assert (
            "pair the branch-specific money or rate output with a separate applicability boolean"
            in prompt
        )
        assert (
            "omit assertions about the branch-specific money or rate output" in prompt
        )
        assert (
            "qualifies its averaging basis with operative parenthetical text" in prompt
        )
        assert (
            "includes periods in which the claimant does no work but disregards other absences"
            in prompt
        )
        assert "generic `average_weekly_income_*` input" in prompt
        assert (
            "`such other payments as may ... enable the claimant's average weekly income to be determined more accurately`"
            in prompt
        )
        assert (
            "do not leave the branch money output unconditionally equal to the input average"
            in prompt
        )
        assert (
            "do not reuse the parent provision's generic final-amount phrase" in prompt
        )
        assert (
            "name the principal money or rate output after this limb's own basis or method"
            in prompt
        )

    def test_build_eval_prompt_for_purpose_limited_deeming_discourages_unsupported_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/9A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of paragraph (9)(b), and for that purpose only, "
                "the amounts specified in paragraph (5) shall be treated as though "
                "they were earnings."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/9A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-9A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "purpose-limited deeming clauses" in prompt
        assert "do not use `status: entity_not_supported`" in prompt
        assert (
            "paragraph-(5) amounts treated as earnings for paragraph-(9)(b) only"
            in prompt
        )

    def test_build_eval_prompt_for_uk_residual_determination_limb_requires_other_case_condition(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the amount to be included in the claimant's weekly income shall be determined—\n\n"
                "(iv)\n\n"
                "in any other case, by multiplying the amount of the payment by 7 and dividing "
                "the product by the number of days in the period in respect of which it is made."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "For residual sibling limbs phrased like `in any other case`" in prompt
        assert "Do not treat the shared parent triggers alone as sufficient" in prompt
        assert "model a local residual-case fact or applicability helper" in prompt
        assert "no more specific sibling case applies" in prompt
        assert (
            "include a case where the parent conditions hold but the residual `other case` condition is false"
            in prompt
        )

    def test_build_eval_prompt_for_shall_be_treated_discourages_fact_input_and_vacuous_true(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "If a claimant is entitled to receive a payment to which paragraph (5) "
                "applies, the amount of that payment shall be treated as if made in "
                "respect of a period of a year."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not introduce a `*_fact` input" in prompt
        assert "do not use vacuous `else: true`" in prompt
        assert (
            "do not replace the amount-level legal effect with a `Person`/`Day` boolean stand-in"
            in prompt
        )
        assert (
            "prefer `status: entity_not_supported` over a pseudo-boolean approximation"
            in prompt
        )
        assert (
            "If the current ontology cannot faithfully tie the deeming effect to the same payment amount"
            in prompt
        )

    def test_build_eval_prompt_for_claimant_incurred_expenses_preserves_claimant_predicate(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/2/f/i",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "travelling expenses incurred by the claimant between his home and place "
                "of employment;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/f/i",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-f-i.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "If an expenses limb says the expenses are `incurred by the claimant`"
            in prompt
        )
        assert "preserve that claimant-incurred predicate explicitly" in prompt
        assert "Do not collapse it into only an employer-made-payment fact" in prompt

    def test_build_eval_prompt_for_claim_date_reference_day_uses_single_operative_date(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of paragraph (2)(b) the last payments are the last payments "
                "before the date the claim was made or treated as made or, if there is a "
                "subsequent supersession under section 10 of the Social Security Act 1998, "
                "the last payments before the date of the supersession."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "preserve a single legally operative reference day" in prompt
        assert "model one canonical operative claim-date fact" in prompt
        assert (
            "do not encode separate `day_is_date_claim_was_made` and `day_is_date_claim_was_treated_as_made` facts and then combine them with `or`"
            in prompt
        )
        assert (
            "include one no-supersession case for the operative claim date and one supersession case for the supersession date"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_override_discourages_permission_gate(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Subject to regulation 17B(6), in the case of any income taken into "
                "account for the purpose of calculating a person's income, there shall "
                "be disregarded any amount payable by way of tax."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "treat the cited provision as a possible override or displacement" in prompt
        )
        assert "model a local override/displacement boolean" in prompt
        assert (
            "Do not encode those `Subject to ...` qualifiers as helper names like `*_permits_*`"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_unavailable_imports_allows_paragraph_specific_inputs(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/2/e",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Subject to paragraphs (3), (4) and (4A), “earnings” in the case of "
                "employment as an employed earner, means any remuneration or profit "
                "derived from that employment and includes any payment by way of a retainer."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/e",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-e.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "When canonical imports for those cited paragraphs are available in the workspace, import them."
            in prompt
        )
        assert (
            "paragraph-specific local inputs are acceptable for an isolated slice artifact"
            in prompt
        )
        assert (
            "preserve the cited paragraph numbers and the branch-specific legal effect"
            in prompt
        )

    def test_build_eval_prompt_for_missing_cross_reference_exception_requires_defer(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 45A(d)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Paragraph (1) shall not apply to a transaction to which section "
                "381(a) applies if the employee continues to be employed by the "
                "acquiring corporation."
            ),
            axiom_rules_path=tmp_path / "rulespec-us",
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(d)",
            "repo-augmented",
            workspace,
            [],
            target_file_name="d.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Missing cited RuleSpec sources detected" in prompt
        assert "`us:statutes/26/381/a`" in prompt
        assert "Do not create local facts such as" in prompt
        assert "`section_381_a...`" in prompt
        assert (
            "emit `module.status: deferred` or `module.status: entity_not_supported`"
            in prompt
        )
        assert "leave any tests" in prompt
        assert "deferred surface empty" in prompt
        assert "module.deferred_outputs[]" in prompt
        assert "absolute `output` and `blocked_by` targets" in prompt
        assert "copied child output" in prompt
        assert "parent composition" in prompt

    def test_build_eval_prompt_for_missing_definition_dependency_requires_defer(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 45A(e)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "The term wages has the same meaning given to such term in "
                "section 51. All employers treated as a single employer under "
                "section 52 shall be treated as a single employer for purposes "
                "of this section."
            ),
            axiom_rules_path=tmp_path / "rulespec-us",
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(e)",
            "repo-augmented",
            workspace,
            [],
            target_file_name="e.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Missing cited RuleSpec sources detected" in prompt
        assert "`us:statutes/26/51`" in prompt
        assert "`us:statutes/26/52`" in prompt
        assert "same-meaning" in prompt
        assert "treated-as" in prompt
        assert (
            "emit `module.status: deferred` or `module.status: entity_not_supported`"
            in prompt
        )
        assert "omit or defer only the" in prompt
        assert "blocked surface" in prompt

    def test_build_eval_prompt_for_proration_tests_prefers_exact_division(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 45A(e)(5)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For any taxable year having less than 12 months, the amount "
                "shall be multiplied by a fraction, the numerator of which is "
                "the number of days in the taxable year and the denominator of "
                "which is 365."
            ),
            axiom_rules_path=tmp_path / "rulespec-us",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(e)(5)",
            "cold",
            workspace,
            [],
            target_file_name="5.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "For proration tests with a source-stated denominator" in prompt
        assert "choose input amounts divisible by that denominator" in prompt
        assert "36500 * 182 / 365 = 18200" in prompt

    def test_build_eval_prompt_for_pure_cross_reference_computation_preserves_distinct_cited_alternatives(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "In the case of the earnings of self-employed earners, the amounts "
                "specified in paragraph (10) shall be taken into account in accordance "
                "with paragraph (4) or, as the case may be, paragraph (10) of regulation "
                "13 of the Computation of Earnings Regulations, as having effect in the "
                "case of state pension credit."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "do not replace the cited computation with local boolean `*_route_is_satisfied` or `*_fact` placeholders"
            in prompt
        )
        assert "do not emit a top-level `status: deferred` stub" in prompt
        assert (
            "do not collapse those cited alternatives into one generic treatment gate"
            in prompt
        )
        assert (
            "preserve the distinct cited alternatives with paragraph-specific imports or local facts/amounts"
            in prompt
        )
        assert "do not invent an extra `no treatment applies` branch" in prompt
        assert (
            "do not make the cited route-selection flags part of whether the paragraph itself applies"
            in prompt
        )
        assert (
            "do not encode the consequence as an unqualified `if paragraph_4_route: paragraph_4_amount else: paragraph_10_amount`"
            in prompt
        )
        assert (
            "Paragraph (10) must be selected by a paragraph-(10) route fact/import or by a derived paragraph-(10) route helper"
            in prompt
        )
        assert "prefer a single mutually exclusive route selector" in prompt
        assert (
            "Do not expose two independent route booleans that allow both routes or neither route to be selected"
            in prompt
        )
        assert "a safe local-placeholder shape is" in prompt
        assert (
            "paragraph-(10) route is derived as the applicable paragraph with not paragraph-(4) route"
            in prompt
        )
        assert "Do not create an invalid-route output branch that returns `0`" in prompt
        assert "self-employed earnings trigger the paragraph" in prompt
        assert (
            "regulation 13 paragraph (4) or paragraph (10) chooses the accounting route"
            in prompt
        )
        assert (
            "avoid a false case that makes a self-employed-earner branch fail merely because neither local route flag was selected"
            in prompt
        )
        assert "include separate cases for the distinct cited alternatives" in prompt

    def test_build_eval_prompt_requires_calendar_date_test_periods(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13A/3/b",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "The amount of the guarantee credit payable in respect of the part-week "
                "shall be determined by multiplying the resulting figure by the number "
                "of days in the part-week."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13A/3/b",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13A-3-b.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Use concrete ISO calendar dates like `2025-03-21`" in prompt
        assert "do not use ISO week strings like `2025-W13`" in prompt

    def test_build_eval_prompt_for_uk_leaf_forbids_speculative_future_period_tests(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "The test file must contain YAML only" in prompt
        assert "must be a YAML list beginning with `- name:` entries" in prompt
        assert "Do not add speculative future-period tests" in prompt
        assert (
            "Use factual predicates or quantities in `input:`, not the output variable being asserted"
            in prompt
        )
        assert "Use concrete scalar values, not formula strings" in prompt
        assert "Use `period`, `input`, and `output` keys" in prompt

    def test_build_eval_prompt_for_uk_branch_leaves_requires_branch_specific_names(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) 332.95 per week in the case of a claimant who has a partner.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "encode the branch identity in the output variable name" in prompt
        assert "principal output variable must encode that deepest token" in prompt
        assert "`standard_minimum_guarantee`" in prompt
        assert "`child_benefit_weekly_rate`" in prompt

    def test_build_eval_prompt_for_where_must_clauses_requires_inapplicable_case(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/4A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the young person is aged 19, he or she must have started the education "
                "or training before reaching that age."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/4A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-4A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Where X, Y must ..." in prompt
        assert "Include a `.test.yaml` case where `X` is false" in prompt

    def test_build_eval_prompt_for_uk_leaf_discourages_opaque_condition_helpers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) ... only person or elder or eldest person ... £26.05.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "avoid opaque placeholders like `*_condition`" in prompt
        assert "`child_benefit_is_only_person`" in prompt
        assert "`claimant_has_partner`" in prompt
        assert "`is_joint_claimant`" in prompt

    def test_build_eval_prompt_for_single_row_fixed_amount_discourages_placeholder_names_and_applies_helpers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2013/376/regulation/36",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-07.\n\n"
                "Structured table:\n"
                "Element | Amount for each assessment period\n"
                "single claimant aged under 25 | £316.98"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2013-376-regulation-36.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Use a descriptive legal variable name" in prompt
        assert "not a path- or source-id-derived placeholder" in prompt
        assert "do not invent a fresh `*_applies` helper" in prompt
        assert "do not invent alternate zero-amount tests" in prompt
        assert "Do not emit `otherwise:`" in prompt
        assert "Do not emit `before YYYY-MM-DD: 0`" in prompt
        assert "Do not emit malformed date blocks like `from 0:`" in prompt
        assert "use boolean or fact-shaped helper inputs" in prompt
        assert "Do not invent sample ages like `2`, `3`, `24`, or `25`" in prompt
        assert "keep `.test.yaml` outputs scalar" in prompt
        assert "keep the row-defining conditions satisfied" in prompt
        assert "principal amount rule should usually be a grounded constant" in prompt
        assert "Do not include `alternate_branch_*` tests" in prompt
        assert "write `2500`, not `2,500`" in prompt

    def test_build_eval_prompt_requires_named_scalars_for_repeated_and_threshold_numbers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/2A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where a person is engaged in employments specified in paragraph 2 but "
                "his earnings are less than £20 and he is also engaged in other employment, "
                "so much of his other earnings as would not exceed £20. "
                "A non-dependant aged 18 or over is treated differently. "
                "See section 3(4)."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/2A",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RuleSpec"
            in prompt
        )
        assert (
            "If the same numeric value appears twice in materially different legal roles"
            in prompt
        )
        assert (
            "reuse that named scalar everywhere the rule compares against or computes with that number"
            in prompt
        )
        assert "Do not simplify source-stated ratios or fractions" in prompt
        assert "ungrounded decimal such as `0.10`" in prompt
        assert (
            'If `./source.txt` says someone is "aged 18 or over", "under 25"' in prompt
        )
        assert "Do not create scalar variables for citation numbers" in prompt
        assert (
            "Do not invent `dtype: String` variables just to restate the effective date"
            in prompt
        )
        assert "Axiom formulas have no date literal type" in prompt
        assert "Do not put the date or year value in the fact name" in prompt
        assert "taxable_year_begins_after_termination_date" in prompt
        assert "`taxable_year_begins_after_2024_and_before_2029` or" in prompt
        assert (
            "Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`" in prompt
        )
        assert "overrides preservation of existing local input names" in prompt
        assert (
            "Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables"
            in prompt
        )
        assert "module.summary` or the rule's proof excerpt" in prompt
        assert "exact source phrase containing that number" in prompt
        assert "`==` for equality" in prompt

    def test_prepare_eval_workspace_injects_resolved_defined_term_stub(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        definition_files = [
            item for item in workspace.context_files if item.kind == "definition_stub"
        ]
        assert len(definition_files) == 1
        assert (
            definition_files[0].workspace_path
            == "context/legislation/ukpga/2002/16/section/3ZA/3.yaml"
        )
        assert (
            definition_files[0].import_path == "legislation/ukpga/2002/16/section/3ZA/3"
        )
        stub_path = workspace.root / definition_files[0].workspace_path
        assert stub_path.exists()
        assert "is_member_of_mixed_age_couple" in stub_path.read_text()

    def test_prepare_eval_workspace_copies_resolved_canonical_concept_file(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us-co"
        concept_file = policy_repo_root / "statutes" / "crs" / "26-2-703" / "12.yaml"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    C.R.S. § 26-2-703(12)
    Definitions

    "Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
rules:
  - name: is_individual_responsibility_contract
    kind: input
    entity: Person
    dtype: Boolean
    period: Month
"""
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            extra_context_paths=[],
        )

        concept_files = [
            item for item in workspace.context_files if item.kind == "canonical_concept"
        ]
        assert len(concept_files) == 1
        assert (
            concept_files[0].workspace_path == "context/statutes/crs/26-2-703/12.yaml"
        )
        assert concept_files[0].import_path == "us-co:statutes/crs/26-2-703/12"
        copied_path = workspace.root / concept_files[0].workspace_path
        assert copied_path.exists()
        assert "is_individual_responsibility_contract" in copied_path.read_text()

    def test_hydrate_eval_root_places_resolved_definition_stub_under_runner_root(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        runner_root = tmp_path / "case" / "openai-gpt-5.4"
        source_dir = runner_root / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "example.yaml").write_text("format: rulespec/v1\nrules: []\n")

        _hydrate_eval_root(runner_root, workspace)

        hydrated = (
            runner_root
            / "legislation"
            / "ukpga"
            / "2002"
            / "16"
            / "section"
            / "3ZA"
            / "3.yaml"
        )
        assert hydrated.exists()
        assert "is_member_of_mixed_age_couple" in hydrated.read_text()

    def test_build_eval_prompt_includes_resolved_defined_term_guidance(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Resolved definition files are available below." in prompt
        assert "mixed-age couple" in prompt
        assert (
            "legislation/ukpga/2002/16/section/3ZA/3#is_member_of_mixed_age_couple"
            in prompt
        )
        assert (
            "import that canonical definition instead of inventing a leaf-local helper"
            in prompt
        )
        assert "Do not replace that import with a local deferred stub" in prompt
        assert (
            "Do not encode such local factual predicates as placeholder constants like `true` or `false`."
            in prompt
        )
        assert (
            "Do not encode such local factual predicates as `status: deferred`"
            in prompt
        )

    def test_build_eval_prompt_includes_resolved_canonical_concept_guidance(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us-co"
        concept_file = policy_repo_root / "statutes" / "crs" / "26-2-703" / "12.yaml"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    C.R.S. § 26-2-703(12)
    Definitions

    "Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
rules:
  - name: is_individual_responsibility_contract
    kind: input
    entity: Person
    dtype: Boolean
    period: Month
"""
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "co/regulation/3.609.1/A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "Resolved canonical concept files from this corpus are available below."
            in prompt
        )
        assert "individual responsibility contract" in prompt
        assert (
            "statutes/crs/26-2-703/12#is_individual_responsibility_contract" in prompt
        )
        assert (
            "import or re-export that exact canonical concept instead of duplicating it locally"
            in prompt
        )

    def test_build_eval_prompt_includes_import_vs_local_helper_protocol(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="26 USC 24(c)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text='The term "qualifying child" means a qualifying child as defined in section 152(c).',
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 24(c)",
            "cold",
            workspace,
            [],
            target_file_name="24-c.yaml",
            include_tests=True,
        )

        assert (
            "emit the upstream import instead of restating the concept locally"
            in prompt
        )
        assert "already executable" in prompt
        assert "do not replace it with" in prompt
        assert "requested source itself defines a legal status or test" in prompt
        assert "IRC section 112" not in prompt
        assert "Hard requirement for IRC section 112" not in prompt
        assert "same concept or output name" in prompt
        assert "directly rounded final amount table" in prompt
        assert "round the" in prompt
        assert "increase before adding it to the base amount" in prompt
        assert "17300, not 17325" in prompt
        assert "Outputs named `taxable_income`" in prompt
        assert "if condition: max(0, branch_a) else: max(0, branch_b)" in prompt
        assert "rate * min(max(0, earned_income), cap)" in prompt
        assert "says a value is determined `in accordance with section X`" in prompt
        assert "do not invent `import` statements or `imports:` blocks" in prompt
        assert "Importing a child rate or threshold is not enough" in prompt
        assert "`to the extent`" in prompt
        assert "all-or-nothing zeroing" in prompt
        assert "current\n  requested source changes the basis" in prompt
        assert "internally handled its own `to the extent` exclusion" in prompt

    def test_build_eval_prompt_highlights_cited_context_import_exports(self, tmp_path):
        policy_repo_root = tmp_path / "rulespec-us"
        cited_file = policy_repo_root / "statutes" / "26" / "1211.yaml"
        cited_file.parent.mkdir(parents=True, exist_ok=True)
        cited_file.write_text(
            """format: rulespec/v1
rules:
  - name: other_taxpayer_capital_losses_allowed
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: allowed_capital_losses
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 1222",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "The term net capital loss means the excess of the losses from "
                "sales or exchanges of capital assets over the sum allowed under "
                "section 1211."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[cited_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 1222",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="1222.yaml",
            target_ref_prefix="us:statutes/26/1222",
            include_tests=True,
        )

        assert "Mandatory cited RuleSpec imports detected from source text" in prompt
        assert "Source cites `1211`" in prompt
        assert "`us:statutes/26/1211#other_taxpayer_capital_losses_allowed`" in prompt
        assert "Do not keep a local `_under_section_...`" in prompt

    def test_build_eval_prompt_treats_in_lieu_citation_as_displaced_context(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us"
        cited_file = policy_repo_root / "statutes" / "26" / "164" / "f.yaml"
        cited_file.parent.mkdir(parents=True)
        cited_file.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_tax_deduction
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: self_employment_tax * 0.5
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 1402(a)(12)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "In lieu of the deduction provided by section 164(f), there "
                "shall be allowed a deduction equal to the product of the "
                "taxpayer's net earnings from self-employment and one-half of "
                "the rates imposed by section 1401."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[cited_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 1402(a)(12)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="12.yaml",
            target_ref_prefix="us:statutes/26/1402/a/12",
            include_tests=True,
        )

        assert "Source cites `164(f)`" in prompt
        assert "displacement or replacement phrase" in prompt
        assert "Do not import the cited final amount" in prompt
        assert "prefer the final imported output" not in prompt

    def test_build_eval_prompt_guides_excluded_child_branch_imports(self, tmp_path):
        policy_repo_root = tmp_path / "rulespec-us"
        parent_file = policy_repo_root / "statutes" / "26" / "1401.yaml"
        child_a_file = policy_repo_root / "statutes" / "26" / "1401" / "a.yaml"
        child_b1_file = policy_repo_root / "statutes" / "26" / "1401" / "b" / "1.yaml"
        parent_file.parent.mkdir(parents=True, exist_ok=True)
        child_a_file.parent.mkdir(parents=True, exist_ok=True)
        child_b1_file.parent.mkdir(parents=True, exist_ok=True)
        parent_file.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: old_age_survivors_and_disability_insurance_tax + self_employment_income_tax + additional_tax
"""
        )
        child_a_file.write_text(
            """format: rulespec/v1
rules:
  - name: old_age_survivors_and_disability_insurance_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: limited_income * rate
"""
        )
        child_b1_file.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_income_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: income * rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 164(f)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "There shall be allowed a deduction equal to one-half of the "
                "taxes imposed by section 1401 (other than the taxes imposed by "
                "section 1401(b)(2))."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[parent_file, child_a_file, child_b1_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 164(f)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="f.yaml",
            target_ref_prefix="us:statutes/26/164/f",
            include_tests=True,
        )

        assert "Excluded cited child branch guidance" in prompt
        assert "excludes `1401(b)(2)`" in prompt
        assert (
            "`us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax`"
            in prompt
        )
        assert "`us:statutes/26/1401/b/1#self_employment_income_tax`" in prompt
        assert "do not import an ancestor aggregate" in prompt

    def test_build_eval_prompt_highlights_terminal_child_exports(self, tmp_path):
        policy_repo_root = tmp_path / "rulespec-us"
        child_file = policy_repo_root / "statutes" / "26" / "3101" / "b" / "2.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: additional_medicare_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2013-01-01'
        formula: 0.009
  - name: additional_medicare_excess_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: max(0, wages - additional_medicare_wage_tax_threshold)
  - name: additional_medicare_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: additional_medicare_excess_wages * additional_medicare_tax_rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 3101",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Section 3101 imposes the taxes described in subsection (b)(2).",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 3101",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="3101.yaml",
            target_ref_prefix="us:statutes/26/3101",
            include_tests=True,
        )

        assert (
            "terminal exports `us:statutes/26/3101/b/2#additional_medicare_tax`"
            in prompt
        )
        assert "Aggregate parent child outputs detected" in prompt
        assert (
            "`us:statutes/26/3101/b/2#additional_medicare_tax`"
            " (derived, Money, TaxUnit)" in prompt
        )
        assert "Do not rebuild a child branch in the parent" in prompt

    def test_build_eval_prompt_requires_child_exception_imports_for_parent_list(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us"
        child_file = (
            policy_repo_root
            / "statutes"
            / "26"
            / "163"
            / "h"
            / "4"
            / "B"
            / "ii"
            / "I.yaml"
        )
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
module:
  summary: Such term shall not include a loan to finance fleet sales.
rules:
  - name: fleet_sales_loan_exception_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
    versions:
      - effective_from: '2025-01-01'
        formula: loan_finances_fleet_sales
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 163(h)(4)(B)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "The term qualified passenger vehicle loan interest means "
                "interest paid on qualifying indebtedness. Such term shall not "
                "include any amount paid or incurred on any of the following:"
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 163(h)(4)(B)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="B.yaml",
            target_ref_prefix="us:statutes/26/163/h/4/B",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Parent exception-list child fragments detected" in prompt
        assert (
            "`us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies`"
            in prompt
        )
        assert "Import each listed child exception output" in prompt
        assert "This overrides the usual small-test-count preference" in prompt
        assert "one blocking companion test" in prompt
        assert "for each listed child exception output" in prompt

    def test_build_eval_prompt_forces_partial_extent_child_parent_defer(self, tmp_path):
        policy_repo_root = tmp_path / "rulespec-us"
        child_file = policy_repo_root / "statutes" / "26" / "3101" / "a.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: oasdi_wage_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1990-01-01'
        formula: 0.062
  - name: oasdi_wage_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: wages * oasdi_wage_tax_rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 3101",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Wages shall be exempt from the taxes imposed by this section "
                "to the extent that such wages are subject exclusively to "
                "another country's social security laws."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 3101",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="3101.yaml",
            target_ref_prefix="us:statutes/26/3101",
            include_tests=True,
        )

        assert "Target-specific schema limit" in prompt
        assert "`us:statutes/26/3101/a#oasdi_wage_tax`" in prompt
        assert "entity_not_supported" in prompt
        assert "`rules: []`" in prompt
        assert "`*_before_exemption`" in prompt

    def test_build_eval_prompt_does_not_defer_parent_for_child_internal_extent(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us"
        child_file = policy_repo_root / "statutes" / "26" / "32" / "c" / "2.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Earned income excludes subsidized work-activity amounts only to the extent
    subsidized under the State program.
rules:
  - name: earned_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 32(c)(2)
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, wages - subsidized_state_work_activity_amounts_received)
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 32",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "(a) In the case of an eligible individual, there shall be "
                "allowed as a credit an amount equal to the credit percentage "
                "of so much of the taxpayer's earned income as does not exceed "
                "the earned income amount. (c)(2) The term earned income means "
                "employee compensation plus self-employment income, and no "
                "amount received for work activities shall be taken into "
                "account, but only to the extent such amount is subsidized "
                "under such State program."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 32",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="32.yaml",
            target_ref_prefix="us:statutes/26/32",
            include_tests=True,
        )

        assert "Target-specific schema limit" not in prompt
        assert "Aggregate parent child outputs detected" in prompt
        assert "`us:statutes/26/32/c/2#earned_income`" in prompt
        assert "internally handled its own `to the extent` exclusion" in prompt

    def test_build_eval_prompt_does_not_defer_amount_adjustment_parent_list(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us-co"
        child_file = policy_repo_root / "statutes" / "39" / "39-22-104" / "4" / "i.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: qualifying_income_subtraction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: income_included_in_federal_taxable_income
"""
        )
        workspace = prepare_eval_workspace(
            citation="us-co/statute/39/39-22-104/4",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(4) There shall be subtracted from federal taxable income:\n"
                "(i) Qualifying income to the extent included in federal "
                "taxable income and exempt from taxes imposed by this article."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "us-co/statute/39/39-22-104/4",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="4.yaml",
            target_ref_prefix="us-co:statutes/39/39-22-104/4",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Target-specific schema limit" not in prompt
        assert "Aggregate parent child outputs detected" in prompt
        assert (
            "`us-co:statutes/39/39-22-104/4/i#qualifying_income_subtraction`" in prompt
        )

    def test_build_eval_prompt_does_not_defer_taxable_income_for_incidental_extent(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us"
        child_file = policy_repo_root / "statutes" / "26" / "63" / "c.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: basic_standard_deduction
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 63",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "Taxable income means gross income minus deductions. Unless an "
                "individual elects to itemize deductions, taxable income means "
                "adjusted gross income minus the standard deduction. Marital "
                "status is determined in accordance with section 7703. The "
                "taxpayer and spouse consent to assessment of any deficiency to "
                "the extent attributable to such change of election."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 63",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="63.yaml",
            target_ref_prefix="us:statutes/26/63",
            include_tests=True,
        )

        assert "Target-specific schema limit" not in prompt
        assert "Taxpayer elections such as electing to itemize deductions" in prompt
        assert "Outputs named `taxable_income`" in prompt

    def test_build_eval_prompt_scopes_partial_extent_to_target_paragraph(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us-co"
        child_file = (
            policy_repo_root / "statutes" / "39" / "39-22-104" / "3" / "p" / "5.yaml"
        )
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: initial_window_addition_to_federal_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2023-01-01'
        formula: federal_deduction_addition
"""
        )
        workspace = prepare_eval_workspace(
            citation="us-co/statute/39/39-22-104/3/p",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(3) There shall be added to federal taxable income:\n"
                "(p) Except as otherwise provided in subsection (3)(p.5), "
                "for income tax years commencing on or after January 1, 2022, "
                "for taxpayers who claim itemized deductions and have federal "
                "adjusted gross income equal to or exceeding four hundred "
                "thousand dollars: (I) For a taxpayer who files a single "
                "return, the amount by which itemized deductions exceed thirty "
                "thousand dollars; and (II) For taxpayers who file a joint "
                "return, the amount by which itemized deductions exceed sixty "
                "thousand dollars.\n"
                "(p.5) For income tax years commencing on or after January 1, "
                "2023, a different addition applies.\n"
                "(4)(i) A subtraction is allowed to the extent included in "
                "federal taxable income."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "us-co/statute/39/39-22-104/3/p",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="p.yaml",
            target_ref_prefix="us-co:statutes/39/39-22-104/3/p",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Target-specific schema limit" not in prompt
        assert "except as otherwise provided in section" in prompt
        assert "If copied context\n  for the cited source is present" in prompt
        assert "do not preserve, rename, or recreate the local" in prompt
        assert "follow the copied-context rule above instead" in prompt

    def test_build_eval_prompt_no_tests_includes_copied_context_boundary_rule(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us-co"
        child_file = (
            policy_repo_root / "statutes" / "39" / "39-22-104" / "3" / "p" / "5.yaml"
        )
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: initial_window_addition_to_federal_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2023-01-01'
        formula: federal_deduction_addition
"""
        )
        workspace = prepare_eval_workspace(
            citation="us-co/statute/39/39-22-104/3/p",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(p) Except as otherwise provided in subsection (3)(p.5), "
                "for taxpayers who claim itemized deductions, the amount by "
                "which itemized deductions exceed thirty thousand dollars."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "us-co/statute/39/39-22-104/3/p",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="p.yaml",
            target_ref_prefix="us-co:statutes/39/39-22-104/3/p",
            runner_backend="openai",
        )

        assert "Test file rules:" not in prompt
        assert "do not preserve, rename, or recreate the local" in prompt
        assert "follow the copied-context rule above instead" in prompt

    def test_build_eval_prompt_scopes_partial_extent_to_numeric_target_paragraph(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us"
        child_file = policy_repo_root / "statutes" / "26" / "999" / "1" / "a.yaml"
        child_file.parent.mkdir(parents=True, exist_ok=True)
        child_file.write_text(
            """format: rulespec/v1
rules:
  - name: child_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_amount * rate
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 999(1)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "(1) First rule imposes a tax.\n"
                "(3) Wages are exempt from taxes imposed by this section "
                "to the extent such wages are subject exclusively to another "
                "country's social security laws."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[child_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 999(1)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="1.yaml",
            target_ref_prefix="us:statutes/26/999/1",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Target-specific schema limit" not in prompt

    def test_build_eval_prompt_recommends_final_deduction_imports(self, tmp_path):
        policy_repo_root = tmp_path / "rulespec-us"
        cited_file = policy_repo_root / "statutes" / "26" / "170" / "p.yaml"
        cited_file.parent.mkdir(parents=True, exist_ok=True)
        cited_file.write_text(
            """format: rulespec/v1
rules:
  - name: nonitemizer_charitable_deduction_cap
    kind: parameter
    dtype: Money
    period: Year
    values:
      2026-01-01: 1000
  - name: nonitemizer_charitable_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: min(charitable_contributions, nonitemizer_charitable_deduction_cap)
"""
        )
        workspace = prepare_eval_workspace(
            citation="26 USC 63",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Taxable income is adjusted gross income minus any deduction "
                "provided in section 170(p)."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[cited_file],
        )

        prompt = _build_eval_prompt(
            "26 USC 63",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="63.yaml",
            target_ref_prefix="us:statutes/26/63",
            include_tests=True,
        )

        assert "For the cited deduction/exemption/credit reference" in prompt
        assert "`us:statutes/26/170/p#nonitemizer_charitable_deduction`" in prompt
        assert "`*_provided_in_section_*`" in prompt

    def test_build_eval_prompt_discourages_fabricated_same_instrument_imports(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/5/a",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) except where paragraph (b) applies, £81.50 per week if paragraph 1(1)(a), (b) or (c) of Part I of Schedule I is satisfied.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6-5-a.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Do not fabricate sibling-file imports" in prompt
        assert "do not guess" in prompt
        assert "do not invent `import` statements or `imports:` blocks" in prompt

    def test_build_eval_prompt_for_openai_inlines_source_text(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "You do not have filesystem tool access in this eval" in prompt
        assert "=== BEGIN SOURCE.TXT ===" in prompt
        assert "Editorial note: current text valid from 2025-04-07." in prompt
        assert "26.05" in prompt

    def test_build_eval_prompt_for_date_silent_source_includes_neutral_scaffold_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(E)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Applications received will be certified for six (6) consecutive months "
                "beginning the first month the assistance unit is found eligible for basic cash assistance."
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(E)",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1-E.yaml",
            include_tests=True,
            runner_backend="codex",
        )

        assert "effective_from: '0001-01-01'" in prompt
        assert "harness-only fallback" in prompt


class TestOpenAIEvalRequest:
    def test_post_openai_eval_request_retries_transient_status(self):
        error_response = Mock()
        error_response.status_code = 502
        ok_response = Mock()
        ok_response.status_code = 200

        with (
            patch("axiom_encode.harness.evals.requests.post") as mock_post,
            patch("axiom_encode.harness.evals.time.sleep"),
        ):
            mock_post.side_effect = [error_response, ok_response]

            response = _post_openai_eval_request(
                headers={"Authorization": "Bearer test"},
                body={"model": "gpt-5.4", "input": "hi"},
            )

        assert response is ok_response
        assert mock_post.call_count == 2

    def test_post_openai_eval_request_retries_request_exception(self):
        ok_response = Mock()
        ok_response.status_code = 200

        with (
            patch("axiom_encode.harness.evals.requests.post") as mock_post,
            patch("axiom_encode.harness.evals.time.sleep"),
        ):
            mock_post.side_effect = [
                requests.exceptions.ReadTimeout("timed out"),
                ok_response,
            ]

            response = _post_openai_eval_request(
                headers={"Authorization": "Bearer test"},
                body={"model": "gpt-5.4", "input": "hi"},
            )

        assert response is ok_response
        assert mock_post.call_count == 2


class TestEvalSuiteManifest:
    def test_load_eval_suite_manifest_supports_policyengine_rule_hint(self, tmp_path):
        manifest_file = tmp_path / "uk-expanded.yaml"
        manifest_file.write_text(
            """
name: UK expanded
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: uc-standard-allowance-single-young
    source_id: uc-std-allowance-single
    corpus_citation_path: us/statute/7/2017
    oracle: policyengine
    policyengine_country: uk
    policyengine_rule_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")

        manifest = load_eval_suite_manifest(manifest_file)

        assert (
            manifest.cases[0].policyengine_rule_hint
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_load_eval_suite_manifest_supports_generalist_review_gate(self, tmp_path):
        manifest_file = tmp_path / "uk-expanded.yaml"
        manifest_file.write_text(
            """
name: UK expanded
runners:
  - openai:gpt-5.4
gates:
  min_generalist_review_pass_rate: 0.95
cases:
  - kind: source
    name: sample
    source_id: sample-source
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.gates.min_generalist_review_pass_rate == 0.95

    def test_run_eval_suite_passes_policyengine_rule_hint_to_source_runner(
        self, tmp_path
    ):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK source suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: uc-standard-allowance-single-young
    source_id: uc-std-allowance-single
    corpus_citation_path: us/statute/7/2017
    oracle: policyengine
    policyengine_country: uk
    policyengine_rule_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result("openai-gpt-5.4", "uc-std-allowance-single")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            return_value=[source_result],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
            )

        assert results == [source_result]
        assert (
            mock_source.call_args.kwargs["policyengine_rule_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_run_eval_suite_records_active_case_before_dispatch(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Active case suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")
        snapshots: list[dict] = []

        def fake_run_source_eval(**_kwargs):
            snapshots.append(json.loads((output_root / "suite-run.json").read_text()))
            return [source_result]

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=fake_run_source_eval,
            ),
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
            )

        assert len(snapshots) == 1
        active_state = snapshots[0]
        assert active_state["status"] == "running"
        assert active_state["completed_cases"] == 0
        assert active_state["result_count"] == 0
        assert active_state["active_case"]["index"] == 1
        assert active_state["active_case"]["name"] == "tanf-slice"
        assert active_state["active_case"]["output_root"] == str(
            output_root / "01-tanf-slice"
        )
        final_state = json.loads((output_root / "suite-run.json").read_text())
        assert final_state["status"] == "completed"
        assert "active_case" not in final_state

    def test_run_eval_suite_routes_source_case_to_enclosing_policy_repo(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us-tn"
        policy_repo.mkdir()
        runtime_axiom_rules = tmp_path / "axiom-rules-engine"
        runtime_axiom_rules.mkdir()
        corpus_path = tmp_path / "axiom-corpus"
        corpus_path.mkdir()
        output_root = tmp_path / "out"

        manifest = EvalSuiteManifest(
            name="TN suite",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="repo-augmented",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="snap-tn-sua",
                    source_id="snap_standard_utility_allowance_tn",
                    corpus_citation_path="us-tn/policy/snap-standard-utility-allowance",
                    mode="repo-augmented",
                )
            ],
        )
        source_result = _fake_eval_result("openai-gpt-5.4", "snap-tn-sua")

        with (
            patch(
                "axiom_encode.harness.evals.resolve_corpus_source_unit",
                return_value=Mock(
                    body="Tennessee source text",
                    citation_path="us-tn/policy/snap-standard-utility-allowance",
                    source="local",
                    requested="us-tn/policy/snap-standard-utility-allowance",
                ),
            ),
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                return_value=[source_result],
            ) as mock_run_source_eval,
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=runtime_axiom_rules,
                corpus_path=corpus_path,
            )

        assert mock_run_source_eval.call_args.kwargs["policy_path"] == policy_repo
        assert mock_run_source_eval.call_args.kwargs["source_metadata_payload"] == {
            "corpus_citation_path": "us-tn/policy/snap-standard-utility-allowance",
            "corpus_source": "local",
            "requested_source": "us-tn/policy/snap-standard-utility-allowance",
        }
        assert (
            mock_run_source_eval.call_args.kwargs["runtime_axiom_rules_path"]
            == runtime_axiom_rules
        )

    def test_run_eval_suite_retries_transient_exception(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=[RuntimeError("stream disconnected"), [source_result]],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
            )

        assert results == [source_result]
        assert mock_source.call_count == 2

    def test_run_eval_suite_retries_error_results(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        failed = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")
        failed.success = False
        failed.error = "Reconnecting..."
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=[[failed], [source_result]],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
            )

        assert results == [source_result]
        assert mock_source.call_count == 2

    def test_run_eval_suite_does_not_retry_compile_failures(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        failed = _fake_eval_result(
            "openai-gpt-5.4",
            "co-tanf-f",
            compile_pass=False,
        )

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            return_value=[failed],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
            )

        assert results == [failed]
        assert mock_source.call_count == 1

    def test_run_eval_suite_stops_after_usage_limit_error(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Usage limit suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: case-one
    source_id: case-one
    corpus_citation_path: us/statute/7/2017
  - kind: source
    name: case-two
    source_id: case-two
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"

        usage_limited = _fake_eval_result("openai-gpt-5.4", "case-one")
        usage_limited.metrics.generalist_review_pass = False
        usage_limited.metrics.generalist_review_score = None
        usage_limited.metrics.generalist_review_issues = [
            "Reviewer CLI exited 1: You've hit your usage limit."
        ]
        second = _fake_eval_result("openai-gpt-5.4", "case-two")

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=[[usage_limited], [second]],
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
            )

        assert results == [usage_limited]
        assert mock_source.call_count == 1
        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == "failed"
        assert "usage limit" in state["error"].lower()
        assert state["completed_cases"] == 1
        lines = (output_root / "suite-results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_run_eval_suite_retries_reviewer_timeout(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Timeout retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: case-one
    source_id: case-one
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)

        timed_out = _fake_eval_result("openai-gpt-5.4", "case-one")
        timed_out.metrics.generalist_review_pass = False
        timed_out.metrics.generalist_review_score = None
        timed_out.metrics.generalist_review_issues = [
            "Reviewer error: Reviewer CLI exited 1: Timeout after 300s"
        ]
        recovered = _fake_eval_result("openai-gpt-5.4", "case-one")

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=[[timed_out], [recovered]],
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
            )

        assert results == [recovered]
        assert mock_source.call_count == 2

    def test_run_eval_suite_resume_skips_completed_cases(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Resume suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: case-one
    source_id: case-one
    corpus_citation_path: us/statute/7/2017
  - kind: source
    name: case-two
    source_id: case-two
    corpus_citation_path: us/statute/7/2017
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"
        output_root.mkdir()

        first = _fake_eval_result("openai-gpt-5.4", "case-one")
        second = _fake_eval_result("openai-gpt-5.4", "case-two")
        (output_root / "suite-run.json").write_text(
            json.dumps(
                {
                    "manifest": {
                        "name": manifest.name,
                        "path": str(manifest.path),
                        "runners": manifest.runners,
                        "effective_runners": manifest.runners,
                    },
                    "status": "running",
                    "started_at": "2026-04-10T16:17:28+00:00",
                    "updated_at": "2026-04-10T16:30:00+00:00",
                    "total_cases": 2,
                    "completed_cases": 1,
                    "result_count": 1,
                    "last_case_name": "case-one",
                }
            )
            + "\n"
        )
        (output_root / "suite-results.jsonl").write_text(
            json.dumps(
                {
                    "case_index": 1,
                    "case_name": "case-one",
                    "case_kind": "source",
                    "result": first.to_dict(),
                },
                sort_keys=True,
            )
            + "\n"
        )

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                return_value=[second],
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=_write_test_corpus_provision(tmp_path),
                resume_existing=True,
            )

        assert [result.citation for result in results] == ["case-one", "case-two"]
        mock_source.assert_called_once()
        assert mock_source.call_args.kwargs["source_id"] == "case-two"
        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == "completed"
        assert state["started_at"] == "2026-04-10T16:17:28+00:00"
        assert state["completed_cases"] == 2
        lines = (output_root / "suite-results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    @pytest.mark.parametrize(
        ("manifest_filename", "expected_corpus_paths"),
        [
            (
                "us_co_colorado_works_seed.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/F",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/G",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/I",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_co_colorado_works_leaf_seed.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/E",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/G",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/I",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/J",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_co_colorado_works_leaf_repair.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/G",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/I",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/J",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_co_colorado_works_leaf_k_repair.yaml",
                ["us-co/regulation/9-ccr-2503-6/3.606.1/K"],
            ),
            (
                "us_co_colorado_works_leaf_h_repair.yaml",
                ["us-co/regulation/9-ccr-2503-6/3.606.1/H"],
            ),
            (
                "us_co_colorado_works_leaf_closeout.yaml",
                [
                    "us-co/regulation/9-ccr-2503-6/3.606.1/H",
                    "us-co/regulation/9-ccr-2503-6/3.606.1/K",
                ],
            ),
            (
                "us_snap_federal_reconstruction_seed.yaml",
                [
                    "us/statute/7/2017/a",
                    "us/statute/7/2017/c/1",
                    "us/statute/7/2017/c/3",
                    "us/guidance/usda/fns/snap-fy2026-cola/page-1",
                ],
            ),
            ("us_snap_federal_c3_repair.yaml", ["us/statute/7/2017/c/3"]),
            (
                "us_snap_fy2026_cola_table_repair.yaml",
                ["us/guidance/usda/fns/snap-fy2026-cola/page-1"],
            ),
            ("us_snap_asset_test_refresh.yaml", ["us/statute/7/2014/g/1"]),
            (
                "us_snap_asset_test_current_effective_refresh.yaml",
                ["us/guidance/usda/fns/snap-fy2026-cola/page-2"],
            ),
            ("us_snap_eligibility_refresh.yaml", ["us/statute/7/2014"]),
            (
                "us_snap_earned_income_deduction_refresh.yaml",
                ["us/statute/7/2014/e/2/B"],
            ),
            (
                "us_snap_net_income_pre_shelter_refresh.yaml",
                ["us/statute/7/2014/e/6/A"],
            ),
            (
                "us_snap_co_self_employment_expense_option_refresh.yaml",
                ["us-co/regulation/10-ccr-2506-1/4.403.11"],
            ),
            (
                "us_snap_co_child_support_deduction_option_refresh.yaml",
                ["us-co/regulation/10-ccr-2506-1/4.407.5"],
            ),
        ],
    )
    def test_repo_benchmark_manifests_are_corpus_backed(
        self,
        manifest_filename,
        expected_corpus_paths,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / manifest_filename
        )

        assert manifest.mode == "repo-augmented"
        assert [
            case.corpus_citation_path for case in manifest.cases
        ] == expected_corpus_paths
        assert all(case.kind == "source" for case in manifest.cases)
        for case in manifest.cases:
            for context_path in case.allow_context:
                assert "sources" not in context_path.parts


class TestReadinessSummary:
    def test_summarize_readiness_applies_suite_gates(self):
        gates = EvalReadinessGates(
            min_cases=3,
            min_success_rate=1.0,
            min_compile_pass_rate=1.0,
            min_ci_pass_rate=1.0,
            min_zero_ungrounded_rate=1.0,
            min_generalist_review_pass_rate=1.0,
            min_policyengine_pass_rate=0.8,
            max_mean_estimated_cost_usd=0.5,
        )
        results = [
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-a",
                compile_pass=True,
                ci_pass=True,
                policyengine_pass=True,
                policyengine_score=1.0,
                estimated_cost_usd=0.20,
            ),
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-b",
                compile_pass=True,
                ci_pass=True,
                generalist_review_pass=False,
                generalist_review_score=4.0,
                policyengine_pass=False,
                policyengine_score=0.5,
                estimated_cost_usd=0.40,
            ),
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-c",
                compile_pass=True,
                ci_pass=True,
                generalist_review_pass=True,
                generalist_review_score=7.5,
                policyengine_pass=None,
                policyengine_score=None,
                estimated_cost_usd=0.30,
            ),
        ]

        summary = summarize_readiness(results, gates)

        assert summary.total_cases == 3
        assert summary.compile_pass_rate == 1.0
        assert summary.ci_pass_rate == 1.0
        assert summary.zero_ungrounded_rate == 1.0
        assert summary.generalist_review_pass_rate == pytest.approx(
            2 / 3, rel=0, abs=1e-6
        )
        assert summary.mean_generalist_review_score == pytest.approx(6.5)
        assert summary.policyengine_case_count == 2
        assert summary.policyengine_pass_rate == 0.5
        assert summary.mean_estimated_cost_usd == 0.3
        assert summary.ready is False
        gate_results = {gate.name: gate for gate in summary.gate_results}
        assert gate_results["min_cases"].passed is True
        assert gate_results["min_generalist_review_pass_rate"].passed is False
        assert gate_results["min_policyengine_pass_rate"].passed is False
        assert gate_results["max_mean_estimated_cost_usd"].passed is True


class TestRepoAugmentedContext:
    def test_prepare_eval_workspace_allows_arbitrary_identifier_with_explicit_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        context_file = (
            repo_root / "rulespec-us" / "statutes" / "26" / "32" / "b" / "2" / "A.yaml"
        )
        context_file.parent.mkdir(parents=True)
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(F)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="F. Determining Eligibility ... 165 345 518",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[context_file],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        assert manifest["context_files"][0]["import_path"] == "us:statutes/26/32/b/2/A"
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_prepare_eval_workspace_copies_existing_corpus_target(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "rulespec-us-ny"
        target_file = (
            policy_repo_root / "regulations" / "18-nycrr" / "387" / "12" / "f.yaml"
        )
        target_file.parent.mkdir(parents=True)
        target_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: existing_provenance\n"
            "    kind: source_relation\n"
        )
        target_file.with_name("f.test.yaml").write_text("- name: existing_case\n")

        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="us-ny/regulation/18-nycrr/387/12/f",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="Existing NY regulation text.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(target_file)]["kind"] == "existing_target"
        assert (
            copied_sources[str(target_file.with_name("f.test.yaml"))]["kind"]
            == "existing_target_test_context"
        )

    def test_select_context_files_excludes_target(self, tmp_path):
        policy_repo_root = tmp_path / "rulespec-us"
        section_dir = policy_repo_root / "statutes" / "26" / "24"
        section_dir.mkdir(parents=True)
        (section_dir / "a.yaml").write_text("target")
        (section_dir / "b.yaml").write_text("sibling b")
        (section_dir / "c.yaml").write_text("sibling c")

        selected = select_context_files("26 USC 24(a)", policy_repo_root)

        assert section_dir / "a.yaml" not in selected
        assert section_dir / "b.yaml" in selected
        assert section_dir / "c.yaml" in selected

    def test_prepare_eval_workspace_writes_manifest_and_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "26" / "24"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["source_text_file"] == "source.txt"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        assert manifest["context_files"][0]["import_path"] == "us:statutes/26/24/b"
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_prepare_eval_workspace_copies_context_companion_tests(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "26" / "24"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_test = statute_root / "b.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: context_case\n  period: 2026-01\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )
        copied_test = (
            workspace.root / copied_sources[str(context_test)]["workspace_path"]
        )
        assert copied_test.read_text() == "- name: context_case\n  period: 2026-01\n"

    def test_prepare_eval_workspace_canonical_concepts_use_absolute_imports_and_tests(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rulespec-us"
        statute_root = policy_repo_root / "statutes" / "26" / "1402"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_test = statute_root / "b.test.yaml"
        context_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    (b) Self-employment income The term "self-employment income" means the net earnings from self-employment; except that the section 1401(a) cap applies.
rules:
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: net_earnings_from_self_employment
"""
        )
        context_test.write_text(
            "- name: context_case\n"
            "  input:\n"
            "    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 1000\n"
            "  output:\n"
            "    us:statutes/26/1402/b#self_employment_income_for_section_1401_a: 923.5\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 1401(a)",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "There shall be imposed on the self-employment income of every "
                "individual a tax equal to 12.4 percent."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            (item["source_path"], item["kind"]): item
            for item in manifest["context_files"]
        }
        canonical_item = copied_sources[(str(context_file), "canonical_concept")]
        assert canonical_item["import_path"] == "us:statutes/26/1402/b"
        assert (
            "`self-employment income` -> import "
            "`us:statutes/26/1402/b#self_employment_income_for_section_1401_a`"
            in canonical_item["label"]
        )
        companion_item = copied_sources[
            (str(context_test), "implementation_test_context")
        ]
        assert companion_item["import_path"] == "us:statutes/26/1402/b.test"
        copied_test = workspace.root / companion_item["workspace_path"]
        assert copied_test.read_text() == context_test.read_text()
        assert not any(
            item["import_path"] == "statutes/26/1402/b"
            for item in manifest["context_files"]
        )

    def test_prepare_eval_workspace_copies_existing_target_file_as_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "rulespec-us"
        policy_repo_root.mkdir(parents=True)
        target = policy_repo_root / "statutes" / "26" / "3111" / "a.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: employer_oasdi_excise_tax\n"
            "    kind: derived\n"
        )
        target_test = target.with_name("a.test.yaml")
        target_test.write_text("- name: existing_case\n  period: 2026-01\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="us/statute/26/3111/a",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="3111(a) imposes 6.2 percent employer OASDI tax.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(target)]["kind"] == "existing_target"
        assert copied_sources[str(target)]["import_path"] == "us:statutes/26/3111/a"
        assert (
            copied_sources[str(target_test)]["kind"] == "existing_target_test_context"
        )

    def test_build_eval_prompt_preserves_existing_executable_surface(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "rulespec-us"
        policy_repo_root.mkdir(parents=True)
        target = policy_repo_root / "statutes" / "26" / "45A" / "a.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: qualified_wages\n"
            "    kind: derived\n"
            "    entity: Employer\n"
            "    dtype: Money\n"
            "    period: Year\n"
        )

        workspace = prepare_eval_workspace(
            citation="us/statute/26/45A/a",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text="The amount of the credit shall be 20 percent of qualified wages.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 45A(a)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="statutes/26/45A/a.yaml",
            target_ref_prefix="us:statutes/26/45A/a",
            include_tests=True,
            runner_backend="openai",
        )

        assert "copied current target files as context" in prompt
        assert "not as backward compatibility contracts" in prompt
        assert "Source-faithful RuleSpec with canonical legal pointers" in prompt
        assert "Never preserve, rename, or recreate a legacy local input" in prompt
        assert "source-stated formula executable" in prompt
        assert "defer only that branch" in prompt
        assert "Do not treat a missing deferred child branch\n  as zero" in prompt
        assert 'excess of" a cap' in prompt
        assert "min(source_amount, cap)" in prompt

    def test_prepare_eval_workspace_adds_same_section_subsection_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "7" / "2015"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "e.yaml"
        context_test = statute_root / "e.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: student_exception_case\n  period: 2026-01\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="7 USC 2015(d)(2)(C)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "A higher education student is ineligible unless the student "
                    "meets the requirements of subsection (e) of this section."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_file)]["import_path"] == "us:statutes/7/2015/e"
        )
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_country_monorepo_child_context(self, tmp_path):
        policy_repo_root = tmp_path / "rulespec-us"
        child_root = policy_repo_root / "us" / "statutes" / "26" / "36B" / "b" / "3"
        child_root.mkdir(parents=True)
        child_file = child_root / "A.yaml"
        child_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: applicable_percentage_income_tier\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Integer\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2024-01-01'\n"
            "        formula: 0\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="26 USC 36B",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="Section 36B defines the premium assistance credit amount.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(child_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(child_file)]["workspace_path"]
            == "context/statutes/26/36B/b/3/A.yaml"
        )
        assert (
            copied_sources[str(child_file)]["import_path"] == "us:statutes/26/36B/b/3/A"
        )

    def test_prepare_eval_workspace_adds_plural_same_section_subsection_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        additions_root = (
            repo_root / "rulespec-us" / "statutes" / "39" / "39-22-104" / "3"
        )
        subtractions_root = (
            repo_root / "rulespec-us" / "statutes" / "39" / "39-22-104" / "4"
        )
        additions_root.mkdir(parents=True)
        subtractions_root.mkdir(parents=True)
        addition_parent = additions_root.with_suffix(".yaml")
        subtraction_parent = subtractions_root.with_suffix(".yaml")
        addition_parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: subsection_3_additions\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
            "    formula: 0\n"
        )
        subtraction_parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: subsection_4_subtractions\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
            "    formula: 0\n"
        )
        for label in "abcdefghij":
            addition_file = additions_root / f"{label}.yaml"
            addition_test = additions_root / f"{label}.test.yaml"
            subtraction_file = subtractions_root / f"{label}.yaml"
            subtraction_test = subtractions_root / f"{label}.test.yaml"
            addition_file.write_text("format: rulespec/v1\nrules: []\n")
            addition_test.write_text(f"- name: addition_{label}_case\n  period: 2026\n")
            subtraction_file.write_text("format: rulespec/v1\nrules: []\n")
            subtraction_test.write_text(
                f"- name: subtraction_{label}_case\n  period: 2026\n"
            )
        addition_file = additions_root / "d.yaml"
        addition_test = additions_root / "d.test.yaml"
        late_addition_file = additions_root / "j.yaml"
        late_addition_test = additions_root / "j.test.yaml"
        subtraction_file = subtractions_root / "a.yaml"
        subtraction_test = subtractions_root / "a.test.yaml"
        late_subtraction_file = subtractions_root / "j.yaml"
        late_subtraction_test = subtractions_root / "j.test.yaml"

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="us-co/statute/39/39-22-104/2",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "Federal taxable income shall be modified as provided in "
                    "subsections (3) and (4) of this section."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(addition_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(addition_parent)]["kind"] == "implementation_precedent"
        )
        assert (
            copied_sources[str(addition_file)]["import_path"]
            == "us:statutes/39/39-22-104/3/d"
        )
        assert (
            copied_sources[str(addition_test)]["kind"] == "implementation_test_context"
        )
        assert (
            copied_sources[str(late_addition_file)]["kind"]
            == "implementation_precedent"
        )
        assert (
            copied_sources[str(late_addition_test)]["kind"]
            == "implementation_test_context"
        )
        assert (
            copied_sources[str(subtraction_file)]["import_path"]
            == "us:statutes/39/39-22-104/4/a"
        )
        assert (
            copied_sources[str(subtraction_parent)]["kind"]
            == "implementation_precedent"
        )
        assert (
            copied_sources[str(subtraction_test)]["kind"]
            == "implementation_test_context"
        )
        assert (
            copied_sources[str(late_subtraction_file)]["kind"]
            == "implementation_precedent"
        )
        assert (
            copied_sources[str(late_subtraction_test)]["kind"]
            == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_same_section_under_subsection_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "26" / "3121"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "y.yaml"
        context_test = statute_root / "y.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: transferred_employee_case\n  period: 2026\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3121(b)(15)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "Service performed in the employ of an international "
                    "organization, except service which constitutes employment "
                    "under subsection (y)."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_file)]["import_path"] == "us:statutes/26/3121/y"
        )
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_nested_same_section_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        subsection_root = repo_root / "rulespec-us" / "statutes" / "26" / "3121" / "a"
        subsection_root.mkdir(parents=True)
        cap_file = subsection_root / "1.yaml"
        domestic_file = subsection_root / "7.yaml"
        domestic_test = subsection_root / "7.test.yaml"
        cap_file.write_text("format: rulespec/v1\nrules: []\n")
        domestic_file.write_text("format: rulespec/v1\nrules: []\n")
        domestic_test.write_text("- name: domestic_service_case\n  period: 2026\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3121(i)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "Wages shall be subject to the provisions of subsection "
                    "(a)(1) of this section. Domestic service described in "
                    "subsection (a)(7)(B) shall be computed to the nearest dollar."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(cap_file)]["kind"] == "implementation_precedent"
        assert copied_sources[str(cap_file)]["import_path"] == "us:statutes/26/3121/a/1"
        assert (
            copied_sources[str(domestic_file)]["import_path"]
            == "us:statutes/26/3121/a/7"
        )
        assert (
            copied_sources[str(domestic_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_cross_section_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        context_root = repo_root / "rulespec-us" / "statutes" / "26" / "104" / "a"
        context_root.mkdir(parents=True)
        context_file = context_root / "4.yaml"
        context_test = context_root / "4.test.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")
        context_test.write_text("- name: service_injury_case\n  period: 2026\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 22",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=(
                    "No reduction shall be made for any amount described in "
                    "section 104(a)(4)."
                ),
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(context_file)]["kind"] == "implementation_precedent"
        assert (
            copied_sources[str(context_file)]["import_path"] == "us:statutes/26/104/a/4"
        )
        assert (
            copied_sources[str(context_test)]["kind"] == "implementation_test_context"
        )

    def test_prepare_eval_workspace_adds_state_regulation_cross_section_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "rulespec-us-co"
        regulations_root = policy_repo_root / "regulations" / "10-ccr-2506-1"
        regulations_root.mkdir(parents=True)
        disqualification_period = regulations_root / "4.803.2.yaml"
        disqualification_period.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: consent_agreement_disqualification_start_deadline_calendar_days\n"
            "    kind: parameter\n"
            "    dtype: Integer\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            "        formula: '30'\n"
        )
        fair_hearing_parent = regulations_root / "4.411.yaml"
        fair_hearing_parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: fair_hearing_request_deadline_days\n"
            "    kind: parameter\n"
            "    dtype: Integer\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            "        formula: '90'\n"
        )
        source_text = (
            "Disqualification shall continue uninterrupted until completed, "
            "regardless of household eligibility, and shall be imposed in "
            "accordance with Section 4.803.2, F unless contrary to the court "
            "order. The household may also request a hearing under Section "
            "4.411.1."
        )

        selected = _select_cross_section_context_files(
            "us-co/regulation/10-ccr-2506-1/4.804.1",
            source_text,
            policy_repo_root,
        )

        assert selected == [disqualification_period, fair_hearing_parent]

        workspace = prepare_eval_workspace(
            citation="us-co/regulation/10-ccr-2506-1/4.804.1",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=source_text,
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            source_metadata_payload={
                "corpus_citation_path": "us-co/regulation/10-ccr-2506-1/4.804.1",
            },
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert (
            copied_sources[str(disqualification_period)]["import_path"]
            == "us-co:regulations/10-ccr-2506-1/4.803.2"
        )
        assert (
            copied_sources[str(fair_hearing_parent)]["import_path"]
            == "us-co:regulations/10-ccr-2506-1/4.411"
        )

        prompt = _build_eval_prompt(
            "us-co/regulation/10-ccr-2506-1/4.804.1",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="4.804.1.yaml",
            target_ref_prefix="us-co:regulations/10-ccr-2506-1/4.804.1",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Mandatory cited RuleSpec imports detected" in prompt
        assert "`us-co:regulations/10-ccr-2506-1/4.803.2`" in prompt
        assert "`us-co:regulations/10-ccr-2506-1/4.411`" in prompt
        assert "Missing cited RuleSpec sources detected" not in prompt
        assert "us:statutes/us-co:regulations" not in prompt

    def test_prepare_eval_workspace_adds_child_context_for_unavailable_cited_parent(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        section_root = repo_root / "rulespec-us" / "statutes" / "26" / "3101"
        section_root.mkdir(parents=True)
        parent = section_root.with_suffix(".yaml")
        parent.write_text(
            "format: rulespec/v1\nmodule:\n  status: entity_not_supported\nrules: []\n"
        )
        oasdi = section_root / "a.yaml"
        oasdi.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: oasdi_wage_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.062'\n"
        )
        hi = section_root / "b" / "1.yaml"
        hi.parent.mkdir()
        hi.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: hospital_insurance_wage_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1986-01-01'\n"
            "        formula: '0.0145'\n"
        )
        source_text = (
            "For purposes of the preceding sentence, the term applicable "
            "percentage means the percentage equal to the sum of the rates "
            "of tax in effect under subsections (a) and (b) of section 3101 "
            "for the calendar year."
        )

        selected = _select_cross_section_context_files(
            "26 USC 3201",
            source_text,
            repo_root / "rulespec-us",
        )

        assert selected == [parent, oasdi, hi]

        runner = parse_runner_spec("openai:gpt-5.5")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3201",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=source_text,
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(parent)]["import_path"] == "us:statutes/26/3101"
        assert copied_sources[str(oasdi)]["import_path"] == "us:statutes/26/3101/a"
        assert copied_sources[str(hi)]["import_path"] == "us:statutes/26/3101/b/1"

    def test_prepare_eval_workspace_adds_child_rate_context_for_exporting_cited_parent(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        section_root = repo_root / "rulespec-us" / "statutes" / "26" / "1401"
        section_root.mkdir(parents=True)
        parent = section_root.with_suffix(".yaml")
        parent.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: self_employment_oasdi_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.124'\n"
        )
        oasdi_rate = section_root / "a" / "rate.yaml"
        oasdi_rate.parent.mkdir(parents=True)
        oasdi_rate.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: old_age_survivors_and_disability_insurance_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.124'\n"
        )
        hi_rate = section_root / "b" / "1" / "rate.yaml"
        hi_rate.parent.mkdir(parents=True)
        hi_rate.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: self_employment_income_tax_rate\n"
            "    kind: parameter\n"
            "    versions:\n"
            "      - effective_from: '1990-01-01'\n"
            "        formula: '0.029'\n"
        )
        source_text = (
            "There shall be allowed a deduction equal to the product of net "
            "earnings and one-half of the sum of the rates imposed by "
            "subsections (a) and (b) of section 1401."
        )

        selected = _select_cross_section_context_files(
            "26 USC 1402(a)(12)",
            source_text,
            repo_root / "rulespec-us",
        )

        assert parent in selected
        assert oasdi_rate in selected
        assert hi_rate in selected

    def test_prepare_eval_workspace_adds_cross_section_list_and_parent_fallback_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        rules_root = repo_root / "rulespec-us" / "statutes" / "26"

        section_911_child = rules_root / "911" / "a.yaml"
        section_911_child.parent.mkdir(parents=True)
        section_911_child.write_text("format: rulespec/v1\nrules: []\n")
        section_931 = rules_root / "931.yaml"
        section_931.write_text("format: rulespec/v1\nrules: []\n")
        section_933 = rules_root / "933.yaml"
        section_933.write_text("format: rulespec/v1\nrules: []\n")
        source_text = (
            "Modified adjusted gross income means adjusted gross income "
            "increased by any amount excluded from gross income under "
            "sections 911, 931, or 933."
        )

        selected = _select_cross_section_context_files(
            "26 USC 151",
            source_text,
            repo_root / "rulespec-us",
        )

        assert section_911_child in selected
        assert section_931 in selected
        assert section_933 in selected

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 151",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=source_text,
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert (
            copied_sources[str(section_911_child)]["kind"] == "implementation_precedent"
        )
        assert (
            copied_sources[str(section_911_child)]["import_path"]
            == "us:statutes/26/911/a"
        )
        assert copied_sources[str(section_931)]["import_path"] == "us:statutes/26/931"
        assert copied_sources[str(section_933)]["import_path"] == "us:statutes/26/933"

    def test_prepare_eval_workspace_adds_cross_section_ancestor_context_for_deep_citation(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        section_3511 = repo_root / "rulespec-us" / "statutes" / "26" / "3511.yaml"
        section_3511.parent.mkdir(parents=True)
        section_3511.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: specified_credit_applies_to_customer_not_cpeo\n"
            "    kind: derived\n"
            "    entity: Employer\n"
            "    dtype: Judgment\n"
            "    period: Year\n"
        )
        source_text = (
            "Any credit allowed under this section shall be treated as a "
            "credit described in section 3511(d)(2)."
        )

        selected = _select_cross_section_context_files(
            "26 USC 3134(i)",
            source_text,
            repo_root / "rulespec-us",
        )

        assert selected == [section_3511]

        runner = parse_runner_spec("codex:gpt-5.5")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 3134(i)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text=source_text,
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        assert copied_sources[str(section_3511)]["kind"] == "implementation_precedent"
        assert copied_sources[str(section_3511)]["import_path"] == "us:statutes/26/3511"

    def test_build_eval_prompt_warns_on_unavailable_cited_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "rulespec-us"
        policy_repo_root.mkdir(parents=True)
        section_152 = policy_repo_root / "statutes" / "26" / "152.yaml"
        section_152.parent.mkdir(parents=True)
        section_152.write_text(
            "format: rulespec/v1\nmodule:\n  status: entity_not_supported\nrules: []\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 151",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text="A dependent is defined in section 152.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 151",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="151.yaml",
            target_ref_prefix="us:statutes/26/151",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Unavailable cited RuleSpec context detected" in prompt
        assert (
            "`us:statutes/26/152` has `module.status: entity_not_supported`" in prompt
        )
        assert "`_under_section_152`" in prompt
        assert "omit or defer only the affected executable surface" in prompt

    def test_build_eval_prompt_warns_on_unavailable_cited_context_ancestors(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "rulespec-us"
        section_408_p_2_a = (
            policy_repo_root / "statutes" / "26" / "408" / "p" / "2" / "A.yaml"
        )
        section_408_p_2_a.parent.mkdir(parents=True)
        section_408_p_2_a.write_text(
            "format: rulespec/v1\nmodule:\n  status: entity_not_supported\nrules: []\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 3121(a)(5)(H)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "under an arrangement to which section 408(p) applies, "
                "other than elective contributions under paragraph (2)(A)(i) thereof"
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 3121(a)(5)(H)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="H.yaml",
            target_ref_prefix="us:statutes/26/3121/a/5/H",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Unavailable cited RuleSpec context detected" in prompt
        assert "`408_p_2_A`, `408_p_2`, `408_p`" in prompt
        assert "`*_to_which_section_408_p_applies`" in prompt

    def test_prepare_eval_workspace_adds_child_fragment_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        child_root = repo_root / "rulespec-us" / "statutes" / "7" / "2015" / "d" / "2"
        child_root.mkdir(parents=True)
        child_files = []
        for fragment in ("A", "B", "C", "D", "E", "F"):
            child_file = child_root / f"{fragment}.yaml"
            child_file.write_text("format: rulespec/v1\nrules: []\n")
            child_files.append(child_file)
        nested_child_file = child_root / "G" / "1.yaml"
        nested_child_file.parent.mkdir()
        nested_child_file.write_text("format: rulespec/v1\nrules: []\n")
        child_files.append(nested_child_file)

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[],
        ):
            workspace = prepare_eval_workspace(
                citation="7 USC 2015(d)(2)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="A person shall be exempt if subparagraphs (A) through (F) apply.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item for item in manifest["context_files"]
        }
        for child_file in child_files:
            assert copied_sources[str(child_file)]["kind"] == "implementation_precedent"
            assert copied_sources[str(child_file)]["import_path"] == "us:" + (
                child_file.relative_to(repo_root / "rulespec-us")
                .with_suffix("")
                .as_posix()
            )

    def test_prepare_eval_workspace_materializes_corpus_source_metadata(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="snap_sua_tn",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="Tennessee source text",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            source_metadata_payload={
                "relations": [
                    {
                        "relation": "sets",
                        "target": "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance",
                        "jurisdiction": "TN",
                    }
                ]
            },
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["source_metadata_file"] == "source-metadata.json"
        assert (
            manifest["source_metadata"]["relations"][0]["target"]
            == "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance"
        )
        assert workspace.source_metadata_file is not None
        assert workspace.source_metadata_file.exists()

    def test_build_eval_prompt_lists_canonical_context_import_target(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        external_file = (
            repo_root
            / "rulespec-us-co"
            / "regulation"
            / "9-CCR-2503-6"
            / "3.606.1"
            / "F.yaml"
        )
        external_file.parent.mkdir(parents=True, exist_ok=True)
        external_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: grant_standard_for_assistance_unit\n"
            "    kind: input\n"
            "    entity: TanfUnit\n"
            "    dtype: Money\n"
            "    period: Month\n"
        )

        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(I)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Deduct the total from step 2, above, from the grant amount.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[external_file],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(I)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="9-CCR-2503-6-3.606.1-I.yaml",
        )

        assert (
            "inspect `context/regulation/9-CCR-2503-6/3.606.1/F.yaml`; "
            "import target `us-co:regulation/9-CCR-2503-6/3.606.1/F`"
        ) in prompt
        expected_hash = (
            "sha256:" + hashlib.sha256(external_file.read_bytes()).hexdigest()
        )
        assert f"context hash `{expected_hash}`" in prompt
        assert (
            "exports `us-co:regulation/9-CCR-2503-6/3.606.1/F#grant_standard_for_assistance_unit`"
            in prompt
        )
        assert "import.output" in prompt
        assert "import.hash" in prompt
        assert "use `hash: sha256:local`" in prompt
        assert "never use `sha256:self`" in prompt
        assert "do not wrap import targets in quotes" in prompt
        assert (
            "Use the listed import target rather than the `./context/...` inspection path"
            in prompt
        )
        assert (
            "do not guess contradictory expectations for those imported values"
            in prompt
        )
        assert (
            "keep `.test.yaml` inputs and expected outputs consistent with the rows visible in that imported file"
            in prompt
        )
        assert (
            "Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0`"
            in prompt
        )
        assert (
            "Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file"
            in prompt
        )
        assert (
            "In formulas, reference imported exports by their bare local rule name"
            in prompt
        )
        assert "import and use the listed exported symbol from that" in prompt
        assert "context instead of creating a local `section_...`" in prompt
        assert (
            "never write an absolute `us:...#rule_name` reference inside a formula"
            in prompt
        )

    def test_build_eval_prompt_flags_child_branch_sibling_name_collisions(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        child_root = repo_root / "rulespec-us" / "statutes" / "7" / "2015" / "d" / "2"
        child_root.mkdir(parents=True)
        for fragment in ("A", "B"):
            child_file = child_root / f"{fragment}.yaml"
            child_file.write_text(
                "format: rulespec/v1\n"
                "rules:\n"
                "  - name: person_exempt_from_paragraph_1_work_requirements\n"
                "    kind: derived\n"
                "    entity: Person\n"
                "    dtype: Judgment\n"
                "    period: Month\n"
            )

        workspace = prepare_eval_workspace(
            citation="7 USC 2015(d)(2)(A)",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "A person otherwise required to comply with paragraph (1) shall be "
                "exempt if the person is subject to and complying with any work "
                "registration requirement under title IV or the Federal-State "
                "unemployment compensation system."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "7 USC 2015(d)(2)(A)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="A.yaml",
            target_ref_prefix="us:statutes/7/2015/d/2/A",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Sibling export naming for this target" in prompt
        assert "`person_exempt_from_paragraph_1_work_requirements`" in prompt
        assert "copied target currently exports invalid colliding names" in prompt
        assert "Do not export any local rule with a copied sibling's name" in prompt
        assert "not the shared parent consequence" in prompt
        assert "treat that name as stale and rename it" in prompt

    def test_build_eval_prompt_qualifies_generic_relation_when_sibling_reserved(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "26"
        statute_root.mkdir(parents=True)
        sibling_file = statute_root / "32.yaml"
        sibling_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: qualifying_child_of_tax_unit\n"
            "    kind: data_relation\n"
            "    data_relation:\n"
            "      predicate: qualifying_child_of_tax_unit\n"
            "      arity: 2\n"
            "      arguments:\n"
            "        - TaxUnit\n"
            "        - Person\n"
        )

        workspace = prepare_eval_workspace(
            citation="26 USC 24",
            runner=parse_runner_spec("openai:gpt-5.5"),
            output_root=tmp_path / "out",
            source_text=(
                "There shall be allowed a credit with respect to each qualifying "
                "child of the taxpayer."
            ),
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[],
        )
        context_file = EvalContextFile(
            source_path=sibling_file,
            workspace_path=Path("context/statutes/26/32.yaml"),
            import_path="us:statutes/26/32",
            kind="implementation_precedent",
        )

        prompt = _build_eval_prompt(
            "26 USC 24",
            "repo-augmented",
            workspace,
            [context_file],
            target_file_name="24.yaml",
            target_ref_prefix="us:statutes/26/24",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Sibling export naming for this target" in prompt
        assert "`qualifying_child_of_tax_unit`" in prompt
        assert "Make the relation source-specific" in prompt
        assert "copied target currently exports invalid colliding names" not in prompt

    def test_hydrate_eval_root_copies_context_into_import_tree(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "26" / "24"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "c.yaml"
        context_file.write_text(
            "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)

        assert (eval_root / "statutes" / "26" / "24" / "c.yaml").read_text() == (
            "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n"
        )

    def test_prepare_eval_workspace_expands_transitive_context_imports(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)

        section_root = repo_root / "rulespec-us" / "statutes" / "26" / "24"
        section_root.mkdir(parents=True)
        aggregator = section_root / "24.yaml"
        aggregator.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - statutes/26/24/a#ctc_allowance\n"
            "  - statutes/26/24/c#qualifying_child_count\n"
            "rules:\n"
            "  - name: section_24_credit\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
        )
        selected = section_root / "c.yaml"
        selected.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/26/24/c/2#ctc_meets_citizenship_requirement\n"
            "  - us:statutes/26/152/c#qualifying_child_of_taxpayer\n"
            "rules:\n"
            "  - name: qualifying_child_count\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Integer\n"
            "    period: Year\n"
        )

        dep_local = section_root / "c" / "2.yaml"
        dep_local.parent.mkdir(parents=True)
        dep_local.write_text("format: rulespec/v1\nrules: []\n")

        dep_cross_section = (
            repo_root / "rulespec-us" / "statutes" / "26" / "152" / "c.yaml"
        )
        dep_cross_section.parent.mkdir(parents=True)
        dep_cross_section.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[aggregator, selected],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item["kind"] for item in manifest["context_files"]
        }

        assert copied_sources[str(selected)] == "implementation_precedent"
        assert copied_sources[str(dep_local)] == "implementation_dependency"
        assert copied_sources[str(dep_cross_section)] == "implementation_dependency"
        assert str(section_root / "a.yaml") not in copied_sources

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)
        assert (
            eval_root / "statutes" / "26" / "24" / "c" / "2.yaml"
        ).read_text() == "format: rulespec/v1\nrules: []\n"
        assert (
            eval_root / "statutes" / "26" / "152" / "c.yaml"
        ).read_text() == "format: rulespec/v1\nrules: []\n"

    def test_build_eval_prompt_flags_existing_target_unresolved_import(self, tmp_path):
        repo_root = tmp_path / "repos"
        rulespec_us = repo_root / "rulespec-us"
        target = rulespec_us / "statutes" / "26" / "63" / "f.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/26/151#exemption_individual_eligible\n"
            "rules:\n"
            "  - name: spouse_aged_additional_amount_person_entitlement\n"
            "    kind: derived\n"
            "    entity: Person\n"
            "    dtype: Judgment\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: |-\n"
            "          spouse_age_before_close_of_taxable_year >= 65\n"
            "          and exemption_individual_eligible\n"
        )

        section_151 = rulespec_us / "statutes" / "26" / "151.yaml"
        section_151.parent.mkdir(parents=True, exist_ok=True)
        section_151.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: taxpayer_exemption_allowed\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Judgment\n"
            "    period: Year\n"
        )

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        source_text = workspace_root / "source.txt"
        source_text.write_text(
            "The taxpayer shall be entitled to an additional amount for the "
            "spouse if an additional exemption is allowable under section 151(b)."
        )
        workspace = EvalWorkspace(
            root=workspace_root,
            source_text_file=source_text,
            manifest_file=workspace_root / "manifest.json",
        )
        context_files = [
            EvalContextFile(
                source_path=str(target),
                workspace_path="context/statutes/26/63/f.yaml",
                import_path="us:statutes/26/63/f",
                kind="existing_target",
            )
        ]

        prompt = _build_eval_prompt(
            "26 USC 63(f)",
            "repo-augmented",
            workspace,
            context_files,
            target_file_name="f.yaml",
            target_ref_prefix="us:statutes/26/63/f",
        )

        assert "Copied existing target fails current RuleSpec validation" in prompt
        assert "us:statutes/26/151#exemption_individual_eligible" in prompt
        assert "does not export `exemption_individual_eligible`" in prompt
        assert "defer the affected executable surface" in prompt

    def test_repo_augmented_context_resolves_statute_prefixed_dependencies(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "7" / "2014"
        statute_root.mkdir(parents=True)

        selected = statute_root / "e.yaml"
        selected.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/7/2014/2014#snap_household_has_elderly_or_disabled_member\n"
            "  - us:statutes/7/2014/d#snap_gross_income\n"
            "rules:\n"
            "  - name: snap_net_income\n"
            "    kind: derived\n"
            "    entity: Household\n"
            "    dtype: Money\n"
            "    period: Month\n"
        )

        section_file = statute_root / "2014.yaml"
        section_file.write_text("format: rulespec/v1\nrules: []\n")
        cross_file = statute_root / "d.yaml"
        cross_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="7 USC 2017(a)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="2017(a) ...",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[selected],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item["kind"] for item in manifest["context_files"]
        }

        assert copied_sources[str(selected)] == "implementation_precedent"
        assert copied_sources[str(section_file)] in {
            "implementation_precedent",
            "implementation_dependency",
        }
        assert copied_sources[str(cross_file)] in {
            "implementation_precedent",
            "implementation_dependency",
        }

    def test_prompt_includes_scaffold_dates_from_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules-engine"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rulespec-us" / "statutes" / "26" / "24"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The threshold is $1,000 and later $2,000.\n"
            "rules:\n"
            "  - name: threshold\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '1998-01-01'\n"
            "        formula: 1000\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 2000\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        prompt = _build_eval_prompt(
            "26 USC 24(a)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            "a.yaml",
        )

        assert "`1998-01-01`" in prompt
        assert "`2018-01-01`" in prompt
        assert "Prefer the earliest scaffold date" in prompt


class TestUnexpectedAccessDetection:
    def test_flags_parent_directory_traversal(self, tmp_path):
        assert _command_looks_out_of_bounds("bash -lc 'find .. -name *.yaml'", tmp_path)

    def test_allows_workspace_paths(self, tmp_path):
        local = tmp_path / "context" / "b.yaml"
        local.parent.mkdir(parents=True)
        local.write_text("format: rulespec/v1\nrules: []\n")


class TestSourceEval:
    def test_run_model_eval_passes_skip_reviewers_to_evaluate_artifact(self, tmp_path):
        corpus_path = _write_test_corpus_provision(
            tmp_path,
            citation_path="us/statute/7/2017/a",
            body="The source amount is 100.",
        )
        policy_repo_root = tmp_path / "rulespec-us"
        policy_repo_root.mkdir()

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: a.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: The source amount is 100.\n"
                "rules:\n"
                "  - name: source_amount\n"
                "    kind: parameter\n"
                "    dtype: Number\n"
                "    versions:\n"
                "      - effective_from: '2025-01-01'\n"
                "        formula: 100\n"
                "=== FILE: a.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    source_amount: 100\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_model_eval(
                citations=["us/statute/7/2017/a"],
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
                corpus_path=corpus_path,
                mode="cold",
                include_tests=True,
                skip_reviewers=True,
            )

        assert mock_evaluate_artifact.call_args.kwargs["skip_reviewers"] is True

    def test_run_source_eval_uses_explicit_context_without_statute_lookup(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "axiom-rules-engine"
        policy_repo_root.mkdir()
        context_file = tmp_path / "examples" / "piecewise.yaml"
        context_file.parent.mkdir(parents=True)
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: 9-CCR-2503-6-3.606.1-F.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: F. Determining Eligibility ...\n"
                "rules:\n"
                "  - name: grant_standard\n"
                "    kind: parameter\n"
                "    entity: TaxUnit\n"
                "    dtype: Money\n"
                "    period: Month\n"
                "    versions:\n"
                "      - effective_from: '2024-07-01'\n"
                "        formula: 165\n"
                "=== FILE: 9-CCR-2503-6-3.606.1-F.test.yaml ===\n"
                "- name: base case\n"
                "  period: 2024-07\n"
                "  input: {}\n"
                "  output:\n"
                "    grant_standard: 165\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None

            mock_evaluate_artifact.return_value = None

            results = run_source_eval(
                source_id="9 CCR 2503-6 3.606.1(F)",
                source_text="F. Determining Eligibility ... 165",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[context_file],
            )

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert Path(result.output_file).exists()
        assert Path(result.output_file).with_suffix(".test.yaml").exists()
        assert result.retrieved_files == [str(context_file)]

        prompt = mock_prompt_eval.call_args.args[2]
        assert ".test.yaml" in prompt
        assert "=== FILE:" in prompt
        assert mock_evaluate_artifact.call_args.kwargs["policy_repo_root"] == (
            policy_repo_root
        )

    def test_run_source_eval_passes_oracle_settings_to_evaluate_artifact(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "axiom-rules-engine"
        policy_repo_root.mkdir()

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2006-965-regulation-2.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: https://www.legislation.gov.uk/uksi/2006/965/regulation/2 states 26.05.\n"
                "rules:\n"
                "  - name: child_benefit_enhanced_rate\n"
                "    kind: parameter\n"
                "    dtype: Money\n"
                "    unit: GBP\n"
                "    versions:\n"
                "      - effective_from: '2025-04-07'\n"
                "        formula: 26.05\n"
                "=== FILE: uksi-2006-965-regulation-2.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    child_benefit_enhanced_rate: 26.05\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_source_eval(
                source_id="uksi/2006/965/regulation/2",
                source_text="26.05",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                mode="cold",
                oracle="policyengine",
                policyengine_country="uk",
                skip_reviewers=True,
            )

        assert mock_evaluate_artifact.call_args.kwargs["oracle"] == "policyengine"
        assert mock_evaluate_artifact.call_args.kwargs["policyengine_country"] == "uk"
        assert mock_evaluate_artifact.call_args.kwargs["skip_reviewers"] is True

    def test_run_source_eval_passes_policyengine_rule_hint_to_evaluate_artifact(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "axiom-rules-engine"
        policy_repo_root.mkdir()

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: The amount is 317.82.\n"
                "rules:\n"
                "  - name: source_row_amount\n"
                "    kind: parameter\n"
                "    dtype: Money\n"
                "    unit: GBP\n"
                "    versions:\n"
                "      - effective_from: '2025-04-07'\n"
                "        formula: 317.82\n"
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    source_row_amount: 317.82\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_source_eval(
                source_id="uksi/2013/376/regulation/36/3",
                source_text="317.82",
                runner_specs=["openai:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                mode="cold",
                oracle="policyengine",
                policyengine_country="uk",
                policyengine_rule_hint="uc_standard_allowance_single_claimant_aged_under_25",
            )

        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_rule_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_build_eval_prompt_includes_policyengine_rule_hint(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2013/376/regulation/36/3",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="317.82",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36/3",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
            policyengine_rule_hint="uc_standard_allowance_single_claimant_aged_under_25",
        )

        assert "uc_standard_allowance_single_claimant_aged_under_25" in prompt
        assert "Keep `.test.yaml` inputs oracle-comparable" in prompt
        assert (
            "Prefer a contemporary monthly `.test.yaml` period like `2022-01` or `2024-01`"
            in prompt
        )
        assert (
            "canonical RuleSpec output whose local name is `uc_standard_allowance_single_claimant_aged_under_25`"
            in prompt
        )
        assert (
            "prefer the oracle's direct component facts over inverted household proxy inputs"
            in prompt
        )
        assert "oracle_inputs.policyengine" in prompt
        assert "equivalent" in prompt
        assert "PolicyEngine-native scenario inputs" in prompt
        assert "assert that canonical copied output" in prompt
        assert "key the test by that id rather than the friendly local name" in prompt
        assert "Key inputs by their resolving legal RuleSpec target too" in prompt
        assert (
            "avoid pre-2015 historical periods that PolicyEngine US cannot evaluate"
            in prompt
        )

    def test_build_eval_prompt_includes_sets_source_metadata_guidance(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="snap_standard_utility_allowance_tn",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="The SUA is $451, effective October 1, 2025.",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            source_metadata_payload={
                "relations": [
                    {
                        "relation": "sets",
                        "target": "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance",
                        "jurisdiction": "TN",
                    }
                ]
            },
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "snap_standard_utility_allowance_tn",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
            policyengine_rule_hint="snap_standard_utility_allowance",
        )

        assert "./source-metadata.json" in prompt
        assert "\n  - relation: sets" not in prompt
        assert '"relation": "sets"' in prompt
        assert "kind: source_relation" in prompt
        assert "source_relation.type" in prompt
        assert "record that legal/provenance edge as a separate" in prompt
        assert "Preserve existing or copied `kind: source_relation` records" in prompt
        assert "source_relation.basis.delegation" in prompt
        assert "mirror the imported file's companion test pattern" in prompt
        assert "Never turn an imported derived rule into a fabricated" in prompt
        assert (
            "Every local executable `kind: derived` or `kind: derived_relation` rule"
            in prompt
        )
        assert "Do not assert raw `kind: parameter` rules directly" in prompt
        assert "Use `holds` and `not_holds` for actual `dtype: Judgment`" in prompt
        assert "Use YAML booleans `true` and `false` for local factual" in prompt
        assert (
            "us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance"
            in prompt
        )
        assert "...#*_applies` or `...#*_uses_*" in prompt
        assert (
            "do not add a top-level `imports:` entry to the absolute canonical target path"
            in prompt
        )
        assert "`*_is_in_state` or `*_is_in_jurisdiction`" in prompt
        assert (
            "use only positive/continuity cases rather than a fabricated out-of-jurisdiction false case"
            in prompt
        )
        assert (
            "encode the canonical boolean slot as a direct dated constant `true` or `false`"
            in prompt
        )
        assert (
            "omit an inapplicable false test unless `./source.txt` itself states a narrower in-jurisdiction condition"
            in prompt
        )

    def test_build_eval_prompt_single_amount_slice_disallows_speculative_future_tests(
        self, tmp_path
    ):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/2005/schedule/2",
            runner=runner,
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-06.\n\n"
                "Structured table:\n"
                "Relevant element | Maximum annual rate\n"
                "Severe disability element | £1734\n"
            ),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/2005/schedule/2",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "For a single fixed-amount source slice, a base case is sufficient."
            in prompt
        )
        assert (
            "For a one-row fixed-amount slice with `period: Year`, a base case is sufficient; do not synthesize an `effective_date_boundary` test."
            in prompt
        )
        assert (
            "Add a later same-amount case only when `./source.txt` explicitly says the amount remains unchanged through that later date."
            in prompt
        )
        assert "Do not add speculative future-period tests" in prompt

    def test_single_amount_slice_detection_excludes_conditional_money_leaf(self):
        assert (
            _is_single_amount_table_slice(
                "£20 is disregarded if the claimant is in receipt of Scottish adult disability living allowance."
            )
            is False
        )

    def test_normalize_nonannual_test_period_value_converts_iso_week_to_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value("2025-W13", date(2025, 3, 21))
            == "2025-03-21"
        )

    def test_normalize_nonannual_test_period_value_bumps_explicit_day_before_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value("2026-04-01", date(2026, 4, 2))
            == "2026-04-02"
        )

    def test_normalize_nonannual_test_period_value_bumps_yaml_date_before_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(date(2026, 4, 1), date(2026, 4, 2))
            == "2026-04-02"
        )

    def test_normalize_nonannual_test_period_value_uses_month_period_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-10-01",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-10"
        )

    def test_normalize_nonannual_test_period_value_preserves_prior_month_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-09",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-09"
        )

    def test_normalize_nonannual_test_period_value_preserves_prior_day_month_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-09-30",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-09"
        )

    def test_allows_relative_workspace_reads(self, tmp_path):
        (tmp_path / "source.txt").write_text("text\n")
        command = "bash -lc 'cat ./source.txt && sed -n \"1,40p\" context/statutes/26/24/b.yaml'"
        assert not _command_looks_out_of_bounds(command, tmp_path)


def _fake_eval_result(
    runner: str,
    citation: str,
    *,
    compile_pass: bool = True,
    ci_pass: bool = True,
    generalist_review_pass: bool | None = True,
    generalist_review_score: float | None = 8.0,
    policyengine_pass: bool | None = None,
    policyengine_score: float | None = None,
    estimated_cost_usd: float | None = 0.25,
    ungrounded_numeric_count: int = 0,
) -> EvalResult:
    return EvalResult(
        citation=citation,
        runner=runner,
        backend="codex",
        model="gpt-5.4",
        mode="cold",
        output_file=f"/tmp/{citation}.yaml",
        trace_file=f"/tmp/{citation}.json",
        context_manifest_file=f"/tmp/{citation}.manifest.json",
        duration_ms=1000,
        success=True,
        error=None,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        reasoning_output_tokens=0,
        estimated_cost_usd=estimated_cost_usd,
        actual_cost_usd=None,
        retrieved_files=[],
        unexpected_accesses=[],
        metrics=EvalArtifactMetrics(
            compile_pass=compile_pass,
            compile_issues=[],
            ci_pass=ci_pass,
            ci_issues=[],
            embedded_source_present=True,
            grounded_numeric_count=1 if ungrounded_numeric_count == 0 else 0,
            ungrounded_numeric_count=ungrounded_numeric_count,
            grounding=[
                GroundingMetric(
                    line=1,
                    raw="26.05",
                    value=26.05,
                    grounded=ungrounded_numeric_count == 0,
                )
            ],
            generalist_review_pass=generalist_review_pass,
            generalist_review_score=generalist_review_score,
            generalist_review_issues=[],
            policyengine_pass=policyengine_pass,
            policyengine_score=policyengine_score,
            policyengine_issues=[],
            taxsim_pass=None,
            taxsim_score=None,
            taxsim_issues=[],
        ),
    )

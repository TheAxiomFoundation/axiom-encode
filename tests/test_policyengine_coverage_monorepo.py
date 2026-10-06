"""Coverage classifier routing for canonical country checkouts.

Coverage accepts only an exact ``rulespec-<country>`` checkout or one of its
direct jurisdiction roots. These tests pin canonical legal-ID derivation and
fail-closed rejection of workspaces, legacy standalone repositories, wrappers,
and symlink aliases.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from axiom_oracles.bridges.coverage import (
    build_policyengine_coverage_report,
)

# An output with no registry mapping stays ``unmapped`` regardless of layout,
# which keeps these assertions independent of the packaged mapping registry.
_UNMAPPED_US_RULESPEC = """format: rulespec/v1
rules:
  - name: brand_new_state_helper_xyz
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: state_input
"""

_KS_TANF_RULESPEC = """format: rulespec/v1
rules:
  - name: ks_tanf_maximum_benefit
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: tanf_basic_allowance + tanf_shelter_allowance
"""

_UT_FEP_TABLE_RULESPEC = """format: rulespec/v1
rules:
  - name: gross_income_test_limit
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 927
  - name: net_income_test_snb
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 796
  - name: fep_fep_tp_max_assistance
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 531
  - name: ut_fep_payment_standard
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: fep_fep_tp_max_assistance
"""

_UT_FEP_R986_239_RULESPEC = """format: rulespec/v1
rules:
  - name: gross_countable_income_snb_percentage_limit
    kind: parameter
    dtype: Rate
    period: Month
    versions:
      - effective_from: '2022-10-24'
        formula: |-
          1.85
  - name: work_expense_allowance_per_employed_person
    kind: parameter
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2022-10-24'
        formula: |-
          100
  - name: financial_assistance_gross_countable_income_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2022-10-24'
        formula: |-
          gross_countable_income_snb_percentage_limit * standard_needs_budget_for_household_size
  - name: financial_assistance_gross_countable_income_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2022-10-24'
        formula: |-
          household_gross_countable_income <= financial_assistance_gross_countable_income_limit
  - name: work_expense_allowance_deduction
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2022-10-24'
        formula: |-
          if financial_assistance_gross_countable_income_eligible: work_expense_allowance_per_employed_person * employed_person_count_in_household_unit else: 0
"""

_UNMAPPED_UK_RULESPEC = """format: rulespec/v1
rules:
  - name: brand_new_local_helper_xyz
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: local_input
"""

_BELGIUM_RULESPEC = """format: rulespec/v1
rules:
  - name: household_benefit_amount
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: household_income * 0
"""

_UK_VAT_RULESPEC = """format: rulespec/v1
rules:
  - name: vat_registration_threshold
    kind: parameter
    versions:
      - effective_from: '2024-04-01'
        formula: 90000
  - name: firm_vat_registered
    kind: derived
    versions:
      - effective_from: '2024-04-01'
        formula: firm_has_active_vat_registration
  - name: net_vat_liability
    kind: derived
    versions:
      - effective_from: '2024-04-01'
        formula: standard_rate_output_vat - recoverable_input_vat
"""

_UK_COMPANIES_ACT_SMALL_COMPANY_RULESPEC = """format: rulespec/v1
rules:
  - name: small_company_annual_turnover_threshold
    kind: parameter
    versions:
      - effective_from: '2025-04-06'
        formula: 15
  - name: small_company_qualifying_conditions_met
    kind: derived
    versions:
      - effective_from: '2025-04-06'
        formula: small_company_turnover_condition_met and small_company_balance_sheet_total_condition_met
  - name: company_qualifies_as_small
    kind: derived
    versions:
      - effective_from: '2025-04-06'
        formula: company_is_in_first_financial_year and small_company_qualifying_conditions_met
"""

_UK_DPA_2018_S157_RULESPEC = """format: rulespec/v1
rules:
  - name: uk_gdpr_higher_maximum_fixed_penalty_amount
    kind: parameter
    versions:
      - effective_from: '2018-05-25'
        formula: 17500000
  - name: uk_gdpr_higher_maximum_penalty_amount
    kind: derived
    versions:
      - effective_from: '2018-05-25'
        formula: max(uk_gdpr_higher_maximum_fixed_penalty_amount, uk_gdpr_higher_maximum_turnover_amount)
  - name: uk_gdpr_higher_maximum_turnover_amount
    kind: derived
    versions:
      - effective_from: '2018-05-25'
        formula: undertaking_total_worldwide_annual_turnover * uk_gdpr_higher_maximum_turnover_percentage
  - name: uk_gdpr_higher_maximum_turnover_percentage
    kind: parameter
    versions:
      - effective_from: '2018-05-25'
        formula: 0.04
  - name: uk_gdpr_standard_maximum_fixed_penalty_amount
    kind: parameter
    versions:
      - effective_from: '2018-05-25'
        formula: 8700000
  - name: uk_gdpr_standard_maximum_penalty_amount
    kind: derived
    versions:
      - effective_from: '2018-05-25'
        formula: max(uk_gdpr_standard_maximum_fixed_penalty_amount, uk_gdpr_standard_maximum_turnover_amount)
  - name: uk_gdpr_standard_maximum_turnover_amount
    kind: derived
    versions:
      - effective_from: '2018-05-25'
        formula: undertaking_total_worldwide_annual_turnover * uk_gdpr_standard_maximum_turnover_percentage
  - name: uk_gdpr_standard_maximum_turnover_percentage
    kind: parameter
    versions:
      - effective_from: '2018-05-25'
        formula: 0.02
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _malformed_doubled_ids(report: dict) -> list[str]:
    """Return IDs whose jurisdiction prefix is doubled (``us:us-...``)."""
    pattern = re.compile(r"^([a-z]{2}):\1[-/]")
    return [
        item["legal_id"] for item in report["items"] if pattern.match(item["legal_id"])
    ]


def test_workspace_root_is_rejected(tmp_path):
    """A workspace containing a checkout is not itself a coverage root."""
    workspace = tmp_path / "workspace"
    _write(
        workspace / "rulespec-us" / "us-al" / "policies" / "dhr" / "poe.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    with pytest.raises(ValueError, match="exact rulespec-<country> checkout"):
        build_policyengine_coverage_report(workspace)


def test_direct_monorepo_root_is_enumerated(tmp_path):
    """``--root <rulespec-us>`` should scan that checkout, not only siblings."""
    root = tmp_path / "rulespec-us"
    _write(root / "us-al" / "policies" / "dhr" / "poe.yaml", _UNMAPPED_US_RULESPEC)

    report = build_policyengine_coverage_report(root)

    assert [item["legal_id"] for item in report["items"]] == [
        "us-al:policies/dhr/poe#brand_new_state_helper_xyz"
    ]
    assert {item["repo"] for item in report["items"]} == {"rulespec-us"}


def test_same_named_workspace_wrapper_does_not_become_country_root(tmp_path):
    """GitHub Actions uses ``.../rulespec-us/rulespec-us``; only the child is a checkout."""
    workspace = tmp_path / "rulespec-us"
    checkout = workspace / "rulespec-us"
    _write(
        checkout / "us-al" / "policies" / "dhr" / "poe.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    with pytest.raises(ValueError, match="exact rulespec-<country> checkout"):
        build_policyengine_coverage_report(workspace)


def test_direct_legacy_root_is_rejected(tmp_path):
    """A standalone per-jurisdiction checkout is not a canonical input."""
    root = tmp_path / "rulespec-us-al"
    _write(root / "policies" / "dhr" / "poe.yaml", _UNMAPPED_US_RULESPEC)

    with pytest.raises(ValueError, match="exact rulespec-<country> checkout"):
        build_policyengine_coverage_report(root)


def test_kansas_tanf_keesm_prefix_is_classified_not_comparable(tmp_path):
    """Kansas KEESM 7411 source helpers are explicit non-comparable TANF slices."""
    root = tmp_path / "rulespec-us"
    _write(
        root / "us-ks" / "policies" / "dcf" / "keesm" / "keesm7410.yaml",
        _KS_TANF_RULESPEC,
    )

    report = build_policyengine_coverage_report(root, program="tanf")

    assert len(report["items"]) == 1
    item = report["items"][0]
    assert item["legal_id"] == (
        "us-ks:policies/dcf/keesm/keesm7410#ks_tanf_maximum_benefit"
    )
    assert item["repo"] == "rulespec-us"
    assert item["status"] == "known_not_comparable"
    assert item["mapping_type"] == "not_comparable"
    assert item["policyengine_variable"] == "ks_tanf_maximum_benefit"


def test_utah_fep_payment_standard_is_comparable_but_table_rows_are_not(tmp_path):
    """Utah DWS Table 1 has one PE oracle surface and three raw table rows."""
    root = tmp_path / "rulespec-us"
    rulespec_path = (
        root
        / "us-ut"
        / "policies"
        / "dws"
        / "eligibility-manual"
        / "table-1-financial-monthly-income-limits.yaml"
    )
    _write(rulespec_path, _UT_FEP_TABLE_RULESPEC)
    _write(
        rulespec_path.with_name("table-1-financial-monthly-income-limits.test.yaml"),
        """- name: Utah FEP payment standard size two
  period: 2026-01
  input:
    us-ut:policies/dws/eligibility-manual/table-1-financial-monthly-income-limits#input.assistance_unit_size: 2
  output:
    us-ut:policies/dws/eligibility-manual/table-1-financial-monthly-income-limits#ut_fep_payment_standard: 531
""",
    )

    report = build_policyengine_coverage_report(root, program="tanf")
    items_by_rule = {item["rule_name"]: item for item in report["items"]}

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 3,
    }

    payment_standard = items_by_rule["ut_fep_payment_standard"]
    assert payment_standard["status"] == "comparable"
    assert payment_standard["mapping_type"] == "direct_variable"
    assert payment_standard["policyengine_variable"] == "ut_fep_payment_standard"
    assert payment_standard["tested"] is True
    assert payment_standard["test_output_count"] == 1

    for rule_name in (
        "gross_income_test_limit",
        "net_income_test_snb",
        "fep_fep_tp_max_assistance",
    ):
        item = items_by_rule[rule_name]
        assert item["status"] == "known_not_comparable"
        assert item["mapping_type"] == "not_comparable"


def test_utah_fep_r986_200_239_gross_income_and_work_expense_mappings(tmp_path):
    """R986-200-239 maps exact PE parameters/predicates and classifies helpers."""
    root = tmp_path / "rulespec-us"
    rulespec_path = (
        root / "us-ut" / "regulations" / "admin-rules" / "r986" / "200" / "239.yaml"
    )
    _write(rulespec_path, _UT_FEP_R986_239_RULESPEC)
    _write(
        rulespec_path.with_name("239.test.yaml"),
        """- name: Utah FEP gross-income limit rate
  period: 2024-01
  input:
    us-ut:regulations/admin-rules/r986/200/239#input.standard_needs_budget_for_household_size: 1000
    us-ut:regulations/admin-rules/r986/200/239#input.household_gross_countable_income: 1850
    us-ut:regulations/admin-rules/r986/200/239#input.employed_person_count_in_household_unit: 2
  output:
    us-ut:regulations/admin-rules/r986/200/239#gross_countable_income_snb_percentage_limit: 1.85
    us-ut:regulations/admin-rules/r986/200/239#financial_assistance_gross_countable_income_eligible: holds
    us-ut:regulations/admin-rules/r986/200/239#work_expense_allowance_per_employed_person: 100
""",
    )

    report = build_policyengine_coverage_report(root, program="tanf")
    items_by_rule = {item["rule_name"]: item for item in report["items"]}

    assert report["status_counts"] == {
        "comparable": 3,
        "known_not_comparable": 2,
    }

    gross_rate = items_by_rule["gross_countable_income_snb_percentage_limit"]
    assert gross_rate["status"] == "comparable"
    assert gross_rate["mapping_type"] == "parameter_value"
    assert (
        gross_rate["policyengine_parameter"]
        == "gov.states.ut.dwf.fep.income.gross_income_limit.rate"
    )
    assert gross_rate["tested"] is True

    gross_eligible = items_by_rule[
        "financial_assistance_gross_countable_income_eligible"
    ]
    assert gross_eligible["status"] == "comparable"
    assert gross_eligible["mapping_type"] == "direct_variable"
    assert gross_eligible["policyengine_variable"] == "ut_fep_gross_income_eligible"
    assert gross_eligible["tested"] is True

    work_allowance = items_by_rule["work_expense_allowance_per_employed_person"]
    assert work_allowance["status"] == "comparable"
    assert work_allowance["mapping_type"] == "parameter_value"
    assert (
        work_allowance["policyengine_parameter"]
        == "gov.states.ut.dwf.fep.income.deductions.work_expense_allowance.amount"
    )
    assert work_allowance["tested"] is True

    for rule_name in (
        "financial_assistance_gross_countable_income_limit",
        "work_expense_allowance_deduction",
    ):
        item = items_by_rule[rule_name]
        assert item["status"] == "known_not_comparable"
        assert item["mapping_type"] == "not_comparable"


def test_medicaid_emergency_exact_mapping_overrides_source_prefix(tmp_path):
    """42 USC 1396b(v)'s final emergency-Medicaid output is comparable while
    neighboring source-level helpers remain explicitly non-comparable."""
    root = tmp_path / "rulespec-us"
    _write(
        root / "us" / "statutes" / "42" / "1396b" / "v.yaml",
        """format: rulespec/v1
rules:
  - name: is_emergency_medicaid_eligible
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: has_emergency_medical_condition
  - name: emergency_medical_condition_exists
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: sudden_onset_condition
""",
    )

    report = build_policyengine_coverage_report(root, program="medicaid")
    items_by_rule = {item["rule_name"]: item for item in report["items"]}

    emergency = items_by_rule["is_emergency_medicaid_eligible"]
    assert emergency["legal_id"] == (
        "us:statutes/42/1396b/v#is_emergency_medicaid_eligible"
    )
    assert emergency["status"] == "comparable"
    assert emergency["mapping_type"] == "direct_variable"
    assert emergency["policyengine_variable"] == "is_emergency_medicaid_eligible"

    helper = items_by_rule["emergency_medical_condition_exists"]
    assert helper["legal_id"] == (
        "us:statutes/42/1396b/v#emergency_medical_condition_exists"
    )
    assert helper["status"] == "known_not_comparable"
    assert helper["mapping_type"] == "not_comparable"


def test_monorepo_country_directory_is_not_doubled(tmp_path):
    """Country-level content in ``rulespec-us/us`` keeps the ``us:`` prefix."""
    root = tmp_path / "rulespec-us"
    _write(
        root / "us" / "statutes" / "26" / "9999.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    report = build_policyengine_coverage_report(root)
    ids = [item["legal_id"] for item in report["items"]]

    assert ids == ["us:statutes/26/9999#brand_new_state_helper_xyz"]
    assert _malformed_doubled_ids(report) == []


def test_monorepo_uk_jurisdiction_directories(tmp_path):
    """``uk`` and ``uk-kingston-upon-thames`` directories keep their prefixes."""
    root = tmp_path / "rulespec-uk"
    _write(
        root
        / "uk-kingston-upon-thames"
        / "policies"
        / "kingston-upon-thames"
        / "council-tax-reduction.yaml",
        _UNMAPPED_UK_RULESPEC,
    )
    _write(
        root / "uk" / "policies" / "govuk" / "child-benefit.yaml",
        _UNMAPPED_UK_RULESPEC,
    )

    report = build_policyengine_coverage_report(root)
    ids = {item["legal_id"] for item in report["items"]}

    assert ids == {
        "uk-kingston-upon-thames:policies/kingston-upon-thames/"
        "council-tax-reduction#brand_new_local_helper_xyz",
        "uk:policies/govuk/child-benefit#brand_new_local_helper_xyz",
    }
    assert _malformed_doubled_ids(report) == []
    repos = {item["repo"] for item in report["items"]}
    assert repos == {"rulespec-uk"}


def test_uk_vat_policy_outputs_are_classified_not_comparable(tmp_path):
    """Firm-level GOV.UK VAT outputs are explicit non-comparable UK surfaces."""
    root = tmp_path / "rulespec-uk"
    _write(
        root / "uk" / "policies" / "govuk" / "vat.yaml",
        _UK_VAT_RULESPEC,
    )

    report = build_policyengine_coverage_report(root)

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    for legal_id in (
        "uk:policies/govuk/vat#vat_registration_threshold",
        "uk:policies/govuk/vat#firm_vat_registered",
        "uk:policies/govuk/vat#net_vat_liability",
    ):
        item = items_by_id[legal_id]
        assert item["repo"] == "rulespec-uk"
        assert item["program"] == "vat"
        assert item["status"] == "known_not_comparable"
        assert item["mapping_type"] == "not_comparable"


def test_uk_companies_act_small_company_outputs_are_not_comparable(tmp_path):
    """Companies Act firm-accounting outputs are outside PolicyEngine UK."""
    root = tmp_path / "rulespec-uk"
    _write(
        root / "uk" / "statutes" / "ukpga" / "2006" / "46" / "382.yaml",
        _UK_COMPANIES_ACT_SMALL_COMPANY_RULESPEC,
    )

    report = build_policyengine_coverage_report(root)

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    for legal_id in (
        "uk:statutes/ukpga/2006/46/382#small_company_annual_turnover_threshold",
        "uk:statutes/ukpga/2006/46/382#small_company_qualifying_conditions_met",
        "uk:statutes/ukpga/2006/46/382#company_qualifies_as_small",
    ):
        item = items_by_id[legal_id]
        assert item["repo"] == "rulespec-uk"
        assert item["program"] == "companies_act"
        assert item["status"] == "known_not_comparable"
        assert item["mapping_type"] == "not_comparable"


def test_uk_dpa_2018_s157_penalty_cap_outputs_are_not_comparable(tmp_path):
    """UK GDPR enforcement-penalty caps are outside PolicyEngine UK."""
    root = tmp_path / "rulespec-uk"
    _write(
        root / "uk" / "statutes" / "ukpga" / "2018" / "12" / "157.yaml",
        _UK_DPA_2018_S157_RULESPEC,
    )

    report = build_policyengine_coverage_report(root)

    assert report["total_outputs"] == 8
    assert report["status_counts"] == {"known_not_comparable": 8}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    for rule_name in (
        "uk_gdpr_higher_maximum_fixed_penalty_amount",
        "uk_gdpr_higher_maximum_penalty_amount",
        "uk_gdpr_higher_maximum_turnover_amount",
        "uk_gdpr_higher_maximum_turnover_percentage",
        "uk_gdpr_standard_maximum_fixed_penalty_amount",
        "uk_gdpr_standard_maximum_penalty_amount",
        "uk_gdpr_standard_maximum_turnover_amount",
        "uk_gdpr_standard_maximum_turnover_percentage",
    ):
        item = items_by_id[f"uk:statutes/ukpga/2018/12/157#{rule_name}"]
        assert item["repo"] == "rulespec-uk"
        assert item["program"] == "data_protection"
        assert item["status"] == "known_not_comparable"
        assert item["mapping_type"] == "not_comparable"


def test_belgium_outputs_are_policyengine_non_comparable(tmp_path):
    """Belgium uses EUROMOD/FANTASI household oracles, not PolicyEngine."""
    root = tmp_path / "rulespec-be"
    jurisdictions = {
        "be": "statutes/cir-92/article-1.yaml",
        "be-bru": "regulations/housing/example.yaml",
        "be-dg": "regulations/family/example.yaml",
        "be-vlg": "regulations/social-protection/example.yaml",
        "be-wal": "regulations/housing/example.yaml",
    }
    for prefix, relative in jurisdictions.items():
        _write(root / prefix / relative, _BELGIUM_RULESPEC)

    report = build_policyengine_coverage_report(root)

    assert report["total_outputs"] == 5
    assert report["status_counts"] == {"known_not_comparable": 5}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    for prefix, relative in jurisdictions.items():
        legal_id = (
            f"{prefix}:{Path(relative).with_suffix('').as_posix()}"
            "#household_benefit_amount"
        )
        item = items_by_id[legal_id]
        assert item["repo"] == "rulespec-be"
        assert item["status"] == "known_not_comparable"
        assert item["mapping_type"] == "not_comparable"
        assert item["candidate_priority"] == "P4"


def test_monorepo_program_directory_is_not_a_jurisdiction(tmp_path):
    """``programs/`` emits program specs without becoming a fake jurisdiction."""
    root = tmp_path / "rulespec-us"
    _write(
        root / "us-al" / "policies" / "dhr" / "poe.yaml",
        _UNMAPPED_US_RULESPEC,
    )
    _write(
        root / "us-al" / "programs" / "snap" / "fy-2026.yaml",
        "program: us-al/snap\noutputs:\n  - snap_eligible\n",
    )

    report = build_policyengine_coverage_report(root)
    ids = [item["legal_id"] for item in report["items"]]

    assert ids == [
        "us-al:policies/dhr/poe#brand_new_state_helper_xyz",
        "us-al:programs/snap/fy-2026#snap_eligible",
    ]
    assert all(not legal_id.startswith("programs:") for legal_id in ids)


def test_fake_monorepo_produces_no_malformed_country_doubled_ids(tmp_path):
    """Separate exact country scans emit no doubled jurisdiction IDs."""
    jurisdiction_files = {
        ("us", "us"): "statutes/26/100.yaml",
        ("us", "us-al"): "policies/dhr/poe/100.yaml",
        ("us", "us-ca"): "regulations/mpp/63-300/1.yaml",
        ("us", "us-ny"): "policies/otda/snap/100.yaml",
        ("us", "us-tx"): "policies/hhsc/snap/100.yaml",
        ("uk", "uk"): "policies/govuk/child-benefit.yaml",
        ("uk", "uk-kingston-upon-thames"): (
            "policies/kingston-upon-thames/council-tax-reduction.yaml"
        ),
    }
    for (country, prefix), rel in jurisdiction_files.items():
        content = (
            _UNMAPPED_UK_RULESPEC if prefix.startswith("uk") else _UNMAPPED_US_RULESPEC
        )
        _write(tmp_path / f"rulespec-{country}" / prefix / rel, content)

    reports = [
        build_policyengine_coverage_report(tmp_path / "rulespec-us"),
        build_policyengine_coverage_report(tmp_path / "rulespec-uk"),
    ]

    assert all(_malformed_doubled_ids(report) == [] for report in reports)
    prefixes = {
        item["legal_id"].split(":", 1)[0]
        for report in reports
        for item in report["items"]
    }
    assert prefixes == {
        "us",
        "us-al",
        "us-ca",
        "us-ny",
        "us-tx",
        "uk",
        "uk-kingston-upon-thames",
    }


def test_symlinked_country_checkout_is_rejected(tmp_path):
    """Coverage does not accept an alias to a country checkout."""
    target = tmp_path / "target" / "rulespec-uk"
    _write(
        target / "uk" / "policies" / "govuk" / "child-benefit.yaml",
        _UNMAPPED_UK_RULESPEC,
    )
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    alias = alias_parent / "rulespec-uk"
    os.symlink(target, alias)

    with pytest.raises(ValueError, match="exact rulespec-<country> checkout"):
        build_policyengine_coverage_report(alias)


def test_direct_jurisdiction_root_is_enumerated(tmp_path):
    """An exact direct child of a country checkout is a valid scan root."""
    jurisdiction_root = tmp_path / "rulespec-us" / "us-zz"
    _write(
        jurisdiction_root / "policies" / "new" / "x.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    report = build_policyengine_coverage_report(jurisdiction_root)

    assert report["status_counts"] == {"unmapped": 1}
    assert report["items"][0]["legal_id"] == (
        "us-zz:policies/new/x#brand_new_state_helper_xyz"
    )
    assert report["items"][0]["repo"] == "rulespec-us"
    assert report["items"][0]["file"] == "policies/new/x.yaml"

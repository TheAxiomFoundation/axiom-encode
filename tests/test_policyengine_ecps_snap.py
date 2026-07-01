import shutil
from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.oracles.policyengine import ecps_snap
from axiom_encode.oracles.policyengine.ecps_snap import (
    COMMON_AXIOM_OUTPUT_ID_BY_LABEL,
    JURISDICTION_CONFIGS,
    Period,
    ProjectedCase,
    add_snapscreener_results,
    all_by_id,
    project_deduction_inputs,
    project_income_resource_inputs,
    project_jurisdiction_household_inputs,
    project_raw_utility_inputs,
    project_utility_allowance_type,
    projected_child_support_payment,
    run_axiom_cases,
    set_input_value,
    uses_household_only_bridge,
)
from axiom_encode.oracles.snapscreener import (
    project_payload,
    run_payloads,
    snapscreener_state_key,
)


def test_set_input_value_updates_every_matching_legal_input():
    inputs = {
        "us:policies/usda/snap/fy-2026-cola/maximum-allotments#input.household_size": 1,
        "us:policies/usda/snap/fy-2026-cola/deductions#input.household_size": 1,
        "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#input.household_size": 1,
    }

    set_input_value(inputs, "household_size", 4)

    assert set(inputs.values()) == {4}


def test_axiom_rules_env_prioritizes_active_rulespec_worktree(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    stale_repo = workspace / "rulespec-us"
    active_repo = tmp_path / "worktrees" / "rulespec-us-medicaid-primary-categories"
    program = active_repo / "us" / "statutes" / "42" / "1396a" / "a" / "10.yaml"
    stale_repo.mkdir(parents=True)
    program.parent.mkdir(parents=True)
    program.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_repo))

    env = ecps_snap.axiom_rules_env(program, workspace)

    roots = [
        Path(root)
        for root in env["AXIOM_RULESPEC_REPO_ROOTS"].split(ecps_snap.os.pathsep)
    ]
    assert roots[0] == active_repo.resolve()
    assert stale_repo.resolve() in roots
    assert roots.index(active_repo.resolve()) < roots.index(stale_repo.resolve())


def test_set_input_value_can_skip_optional_unknown_inputs():
    inputs = {
        "us-co:regulations/10-ccr-2506-1/4.407.3#input.verified_higher_homeless_shelter_costs": False,
    }

    set_input_value(
        inputs,
        "snap_claimed_homeless_shelter_deduction",
        0,
        required=False,
    )

    assert "snap_claimed_homeless_shelter_deduction" not in inputs
    assert (
        "us:regulations/7-cfr/273/10#input.snap_claimed_homeless_shelter_deduction"
        not in inputs
    )


def test_optional_household_size_projection_updates_only_when_present():
    inputs = {
        "us-az:policies/des/faa5/na-eligibility-and-benefit-determination/"
        "fy-2026-benefit-calculation#input.na_net_income": 0,
    }

    set_input_value(inputs, "household_size", 3, required=False)

    assert "household_size" not in inputs

    inputs[
        "us:policies/usda/snap/fy-2026-cola/maximum-allotments#input.household_size"
    ] = 1
    set_input_value(inputs, "household_size", 3, required=False)

    assert (
        inputs[
            "us:policies/usda/snap/fy-2026-cola/maximum-allotments#input.household_size"
        ]
        == 3
    )


def test_optional_projection_fields_do_not_create_bare_inputs():
    inputs = {
        "us-az:policies/des/faa5/na-eligibility-and-benefit-determination/"
        "fy-2026-benefit-calculation#input.na_net_income": 0,
    }

    set_input_value(inputs, "household_shelter_costs_incurred", 1200, required=False)

    assert "household_shelter_costs_incurred" not in inputs


def test_all_by_id_requires_every_member_to_hold():
    result = all_by_id(
        [1, 1, 2, 2, 3],
        [True, True, True, False, True],
    )

    assert result == {1: True, 2: False, 3: True}


def test_new_york_projector_uses_federal_income_and_resource_inputs():
    values = {
        "snap_earned_income": [123.45],
        "snap_unearned_income": [67.89],
        "snap_assets": [999],
    }

    projected = project_income_resource_inputs(JURISDICTION_CONFIGS["us-ny"], values, 0)

    assert projected == {
        "snap_countable_earned_income": 123.45,
        "snap_countable_unearned_income": 67.89,
        "snap_gross_monthly_earned_income": 123.45,
        "snap_total_monthly_unearned_income": 67.89,
        "snap_income_exclusions": 0,
        "snap_countable_financial_resources": 999,
    }


def test_new_york_projector_uses_repaired_work_disqualification_input():
    values = {
        "snap_dependent_care_deduction": [0],
        "snap_earned_income": [0],
        "meets_snap_work_requirements": [False],
    }

    projected = project_jurisdiction_household_inputs(
        JURISDICTION_CONFIGS["us-ny"], values, 0
    )

    assert "household_member_failed_snap_work_requirements" not in projected
    assert (
        projected[
            "household_member_disqualified_for_failure_to_comply_with_work_requirements"
        ]
        is True
    )


def test_oregon_utah_bridge_projector_includes_categorical_minimum_signal():
    values = {
        "snap_dependent_care_deduction": [0],
        "snap_net_income": [800],
        "is_snap_eligible": [True],
        "meets_snap_categorical_eligibility": [True],
        "snap_max_allotment": [546],
        "snap_min_allotment": [24],
        "snap_excess_shelter_expense_deduction": [150],
    }

    projected = project_jurisdiction_household_inputs(
        JURISDICTION_CONFIGS["us-ut"], values, 0
    )

    assert projected == {
        "projected_snap_net_income": 800,
        "projected_snap_eligible": True,
        "projected_snap_categorically_eligible": True,
        "projected_snap_maximum_allotment": 546,
        "projected_snap_minimum_allotment": 24,
        "projected_snap_excess_shelter_deduction": 150,
    }


def test_california_projectors_use_california_snap_input_surface():
    config = JURISDICTION_CONFIGS["us-ca"]
    values = {
        "snap_earned_income": [123.45],
        "snap_unearned_income": [67.89],
        "snap_assets": [999],
        "meets_snap_categorical_eligibility": [True],
        "heating_cooling_expense": [10],
    }

    assert project_income_resource_inputs(config, values, 0) == {
        "snap_gross_monthly_earned_income": 123.45,
        "snap_total_monthly_unearned_income": 67.89,
        "snap_countable_financial_resources": 999,
        "snap_categorically_eligible_for_resource_exemption": True,
    }
    assert project_deduction_inputs(
        config,
        dependent_care_deduction=12,
        child_support_deduction=34,
        medical_deduction=200,
    ) == {
        "dependent_care_deduction": 12,
        "child_support_deduction": 34,
        "medical_deduction": 200,
        "household_entitled_to_excess_medical_deduction": True,
        "snap_allowable_monthly_dependent_care_expenses": 12,
        "snap_allowable_monthly_child_support_payments": 34,
        "snap_total_medical_expenses": 235,
    }
    assert project_raw_utility_inputs(config, values, 0, "") == {
        "household_has_heating_and_cooling_costs_separate_from_rent_or_mortgage": True
    }
    assert project_utility_allowance_type(config, "SUA", "") == {
        "household_has_heating_and_cooling_costs_separate_from_rent_or_mortgage": True
    }
    assert project_utility_allowance_type(config, "LUA", "") == {
        "household_has_heating_and_cooling_costs_separate_from_rent_or_mortgage": False
    }
    assert ecps_snap.medical_expenses_for_deduction(150) == 185


def test_run_axiom_cases_uses_configured_california_member_entity(
    monkeypatch, tmp_path
):
    runtime_requests = []

    def fake_run(cmd, **kwargs):
        assert "run-compiled" in cmd
        runtime_requests.append(ecps_snap.json.loads(kwargs["input"]))
        return ecps_snap.subprocess.CompletedProcess(
            cmd,
            0,
            stdout=ecps_snap.json.dumps({"results": [{"outputs": {}}]}),
            stderr="",
        )

    monkeypatch.setattr(ecps_snap.subprocess, "run", fake_run)
    period = Period(
        label="2026-01",
        year=2026,
        month=1,
        start=date(2026, 1, 1),
        end=date(2026, 1, 31),
    )
    config = JURISDICTION_CONFIGS["us-ca"]

    run_axiom_cases(
        binary=tmp_path / "axiom-rules-engine",
        artifact=tmp_path / "program.json",
        cases=[
            ProjectedCase(
                spm_unit_id=42,
                household_id=420,
                inputs={"household-input": 1},
                member_inputs=[{"member-input": True}],
                pe_outputs={},
            )
        ],
        period=period,
        output_ids=["output-id"],
        relation_id=config.relation_id,
        additional_relation_ids=config.additional_relation_ids,
        member_entity_type=config.member_entity_type,
        env={},
    )

    assert config.member_entity_type == "Person"
    request = runtime_requests[0]
    assert request["dataset"]["relations"] == [
        {
            "name": config.relation_id,
            "tuple": ["spm-42-member-1", "spm-42"],
            "interval": {"start": "2026-01-01", "end": "2026-01-31"},
        },
        {
            "name": "us:statutes/7/2012/j#relation.member_of_household",
            "tuple": ["spm-42-member-1", "spm-42"],
            "interval": {"start": "2026-01-01", "end": "2026-01-31"},
        },
    ]
    inputs_by_name = {
        input_item["name"]: input_item for input_item in request["dataset"]["inputs"]
    }
    assert inputs_by_name["household-input"]["entity"] == "Household"
    assert inputs_by_name["member-input"]["entity"] == "Person"


def test_run_axiom_cases_allows_household_only_projection(monkeypatch, tmp_path):
    runtime_requests = []

    def fake_run(cmd, **kwargs):
        runtime_requests.append(ecps_snap.json.loads(kwargs["input"]))
        return ecps_snap.subprocess.CompletedProcess(
            cmd,
            0,
            stdout=ecps_snap.json.dumps({"results": [{"outputs": {}}]}),
            stderr="",
        )

    monkeypatch.setattr(ecps_snap.subprocess, "run", fake_run)
    period = Period(
        label="2026-01",
        year=2026,
        month=1,
        start=date(2026, 1, 1),
        end=date(2026, 1, 31),
    )
    config = JURISDICTION_CONFIGS["us-az"]

    run_axiom_cases(
        binary=tmp_path / "axiom-rules-engine",
        artifact=tmp_path / "program.json",
        cases=[
            ProjectedCase(
                spm_unit_id=42,
                household_id=420,
                inputs={"household-input": 1},
                member_inputs=[],
                pe_outputs={},
            )
        ],
        period=period,
        output_ids=["output-id"],
        relation_id=config.relation_id,
        additional_relation_ids=config.additional_relation_ids,
        member_entity_type=config.member_entity_type,
        env={},
    )

    request = runtime_requests[0]
    assert request["dataset"]["relations"] == []
    assert request["dataset"]["inputs"] == [
        {
            "name": "household-input",
            "entity": "Household",
            "entity_id": "spm-42",
            "interval": {"start": "2026-01-01", "end": "2026-01-31"},
            "value": {"kind": "integer", "value": 1},
        }
    ]


def test_common_snap_outputs_track_current_federal_rulespec_surface():
    assert COMMON_AXIOM_OUTPUT_ID_BY_LABEL["snap_gross_monthly_income"] == (
        "us:regulations/7-cfr/273/10#snap_total_gross_income"
    )
    assert COMMON_AXIOM_OUTPUT_ID_BY_LABEL["snap_excess_shelter_deduction"] == (
        "us:regulations/7-cfr/273/10#snap_excess_shelter_deduction_for_net_income"
    )


def test_snap_ecps_parser_includes_oregon_and_utah():
    parser = ArgumentParser()
    ecps_snap.configure_parser(parser)

    parsed = parser.parse_args(["--jurisdiction", "us-or"])
    assert parsed.jurisdiction == "us-or"
    parsed = parser.parse_args(["--jurisdiction", "us-ut"])
    assert parsed.jurisdiction == "us-ut"


def test_oregon_and_utah_configs_point_to_expected_program_modules():
    oregon = JURISDICTION_CONFIGS["us-or"]
    utah = JURISDICTION_CONFIGS["us-ut"]

    assert oregon.state_code == "OR"
    assert oregon.repo_name == "rulespec-us-or"
    assert oregon.program_relative_path.as_posix() == (
        "policies/odhs/open/fy-2026-benefit-calculation.yaml"
    )
    assert oregon.output_id_by_label["snap_regular_month_allotment"] == (
        "us-or:policies/odhs/open/fy-2026-benefit-calculation"
        "#snap_regular_month_allotment"
    )
    assert utah.state_code == "UT"
    assert utah.repo_name == "rulespec-us-ut"
    assert utah.program_relative_path.as_posix() == (
        "policies/dws/eligibility-manual/fy-2026-benefit-calculation.yaml"
    )
    assert utah.output_id_by_label["snap_eligible"] == (
        "us-ut:policies/dws/eligibility-manual/fy-2026-benefit-calculation"
        "#snap_eligible"
    )


def test_household_only_snap_bridge_jurisdictions_do_not_project_member_inputs():
    assert uses_household_only_bridge(JURISDICTION_CONFIGS["us-az"])
    assert uses_household_only_bridge(JURISDICTION_CONFIGS["us-or"])
    assert uses_household_only_bridge(JURISDICTION_CONFIGS["us-ut"])
    assert not uses_household_only_bridge(JURISDICTION_CONFIGS["us-co"])
    assert not uses_household_only_bridge(JURISDICTION_CONFIGS["us-ny"])


def test_snap_ecps_main_fails_before_policyengine_when_program_module_missing(
    tmp_path,
):
    args = Namespace(
        jurisdiction="us-or",
        workspace_root=tmp_path,
        program=None,
        test_template=None,
        axiom_binary=None,
        state=None,
        year=2026,
        month=1,
        sample_size=1,
        positive_snap_only=False,
        utility_projection="raw-expenses",
        tolerance=1.5,
        max_differences=20,
        fail_on_mismatch=False,
        min_match_rate=None,
        write_csv=None,
        external_oracle=[],
        snapscreener_api_js=None,
        snapscreener_cache_dir=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        ecps_snap.main(args)

    message = str(exc_info.value)
    assert "Oregon SNAP ECPS comparison is configured" in message
    assert "program module:" in message
    assert "fy-2026-benefit-calculation.yaml" in message


def test_colorado_snap_outputs_use_composed_allotment_and_cfr_net_income():
    config = JURISDICTION_CONFIGS["us-co"]
    outputs = config.output_id_by_label

    assert outputs["snap_regular_month_allotment"] == (
        "us-co:regulations/10-ccr-2506-1/4.207.2#snap_allotment"
    )
    assert outputs["snap_net_income"] == (
        "us:regulations/7-cfr/273/10#snap_net_monthly_income"
    )
    assert config.relation_id == "us:statutes/7/2012/j#relation.member_of_household"
    assert config.additional_relation_ids == ()


def test_projected_child_support_includes_gross_income_deduction():
    values = {
        "snap_child_support_deduction": [25],
        "snap_child_support_gross_income_deduction": [100],
    }

    assert projected_child_support_payment(values, 0) == 125


def test_new_york_policyengine_utility_type_projection_sets_region_and_bua():
    projected = project_utility_allowance_type(
        JURISDICTION_CONFIGS["us-ny"], "BUA", "NY_NAS"
    )

    assert projected["household_resides_in_new_york_city"] is False
    assert projected["household_resides_in_nassau_or_suffolk_county"] is True
    assert (
        projected["household_billed_separately_for_non_telephone_standard_utility"]
        is True
    )
    assert (
        projected[
            "household_incurred_or_anticipated_heating_or_cooling_costs_separate_from_rent_or_mortgage"
        ]
        is False
    )


def test_snapscreener_new_york_state_key_uses_dependent_care_first():
    state_key = snapscreener_state_key(
        "NY",
        {
            "snap_utility_region_str": "NY_NAS",
            "snap_dependent_care_deduction": 12,
            "snap_earned_income": 500,
            "has_usda_elderly_disabled": False,
        },
    )

    assert state_key == "NY_NAS_DC"


def test_snapscreener_new_york_state_key_uses_earned_income_tier():
    state_key = snapscreener_state_key(
        "NY",
        {
            "snap_utility_region_str": "NY_NYC",
            "snap_dependent_care_deduction": 0,
            "snap_earned_income": 500,
            "has_usda_elderly_disabled": False,
        },
    )

    assert state_key == "NY_NYC_EI"


def test_snapscreener_payload_projects_ecps_case():
    case = SimpleNamespace(
        pe_outputs={
            "snap_unit_size": 2,
            "snap_utility_region_str": "NY_ONY",
            "snap_utility_allowance_type": "SUA",
            "snap_earned_income": 0,
            "snap_unearned_income": 2400,
            "snap_assets": float("inf"),
            "snap_dependent_care_deduction": 0,
            "snap_excess_medical_expense_deduction": 100,
            "snap_child_support_deduction": 25,
            "snap_child_support_gross_income_deduction": 10,
            "housing_cost": 900,
            "has_usda_elderly_disabled": True,
        }
    )

    payload = project_payload(case, state="NY")

    assert payload["state_or_territory"] == "NY_ONY_DC"
    assert payload["household_size"] == 2
    assert payload["monthly_non_job_income"] == 2400
    assert payload["resources"] == 0
    assert payload["medical_expenses_for_elderly_or_disabled"] == 135
    assert payload["court_ordered_child_support_payments"] == 35
    assert payload["utility_heating"] is True


def test_snapscreener_results_mark_primary_ineligible_rows_noncomparable():
    rows = [
        {
            "pe_snap": 0.0,
            "axiom_snap_allotment": 0.0,
            "pe_snap_eligible": False,
            "axiom_snap_eligible": "not_holds",
        }
    ]

    add_snapscreener_results(
        rows,
        [{"state_or_territory": "NY_ONY_DC"}],
        [
            {
                "status": "OK",
                "estimated_eligibility": True,
                "estimated_monthly_benefit": 298,
            }
        ],
        tolerance=1.5,
    )

    assert rows[0]["snapscreener_comparable"] is False
    assert rows[0]["axiom_snapscreener_match"] is False


def test_snapscreener_runner_executes_public_api_shape(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    api_js = tmp_path / "api.js"
    api_js.write_text(
        """
var SnapAPI = {
  SnapEstimateEntrypoint: class {
    constructor(payload) { this.payload = payload; }
    calculate() {
      return {
        status: "OK",
        estimated_eligibility: this.payload.state_or_territory.endsWith("_DC"),
        estimated_monthly_benefit: this.payload.household_size + 10,
        gross_income_result: this.payload.monthly_job_income + this.payload.monthly_non_job_income,
        net_income_result: 42
      };
    }
  }
};
"""
    )

    results = run_payloads(
        [
            {
                "state_or_territory": "NY_NYC_DC",
                "household_size": 2,
                "monthly_job_income": 100,
                "monthly_non_job_income": 50,
            }
        ],
        api_js=api_js,
    )

    assert results == [
        {
            "status": "OK",
            "estimated_eligibility": True,
            "estimated_monthly_benefit": 12,
            "gross_income_result": 150,
            "net_income_result": 42,
            "errors": [],
        }
    ]

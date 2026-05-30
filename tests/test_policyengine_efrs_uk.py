import argparse
from pathlib import Path

import pytest

import axiom_encode.oracles.policyengine.efrs_uk as efrs_uk
from axiom_encode.oracles.policyengine.efrs_uk import (
    CHILD_BENEFIT_BASE,
    CHILD_BENEFIT_OUTPUTS,
    PERSONAL_ALLOWANCE_BASE,
    PERSONAL_ALLOWANCE_OUTPUTS,
    PERSONAL_ALLOWANCE_PROGRAM_PATH,
    WEEKS_IN_YEAR,
    build_child_benefit_request,
    build_personal_allowance_request,
    compare_outputs,
    compare_uk_efrs,
    policyengine_person_variables_for_surfaces,
    project_child_benefit_inputs,
    project_personal_allowance_inputs,
    select_person_indices,
)


def decimal_output(value):
    return {"value": {"value": str(value)}}


def test_personal_allowance_projection_matches_policyengine_gift_aid_taper():
    projected = project_personal_allowance_inputs(
        {
            "adjusted_net_income": 101_000,
            "gift_aid_grossed_up": 600,
        }
    )

    assert projected == {
        "individual_makes_claim": True,
        "individual_meets_requirements_under_section_56": True,
        "adjusted_net_income": 100_400,
    }


def test_personal_allowance_request_projects_efrs_people():
    request = build_personal_allowance_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "adjusted_net_income": 20_000,
                    "gift_aid_grossed_up": 0,
                    "personal_allowance": 12_570,
                }
            ],
            "person_ids": [7],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": [PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"]["axiom"]],
        }
    ]
    assert {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    } == {
        f"{PERSONAL_ALLOWANCE_BASE}#input.individual_makes_claim": {
            "kind": "bool",
            "value": True,
        },
        f"{PERSONAL_ALLOWANCE_BASE}#input.individual_meets_requirements_under_section_56": {
            "kind": "bool",
            "value": True,
        },
        f"{PERSONAL_ALLOWANCE_BASE}#input.adjusted_net_income": {
            "kind": "decimal",
            "value": "20000.0",
        },
    }


def test_child_benefit_projection_uses_child_index():
    projected = project_child_benefit_inputs(
        {
            "child_benefit_child_index": 2,
            "child_benefit_respective_amount": 17.90 * WEEKS_IN_YEAR,
        }
    )

    assert projected == {
        "during_subsistence_of_marriage_any_party_married_to_more_than_one_person": False,
        "marriage_ceremony_took_place_under_law_permitting_polygamy": False,
        "specified_benefit_allowance_or_increase_paid_for_week_to_person": False,
        "specified_benefit_is_in_respect_of_only_elder_or_eldest_child_for_child_benefit_entitlement": False,
        "child_or_qualifying_young_person_is_only_elder_or_eldest_for_payee": False,
        "paragraph_2_relationship_coordination_applies": False,
        "child_or_qualifying_young_person_is_elder_or_eldest_among_paragraph_2_children": False,
        "payee_is_voluntary_organisation": False,
        "payee_resides_with_parent_otherwise_than_paragraph_2_a": False,
    }


def test_child_benefit_request_filters_to_positive_respective_amount_rows():
    request = build_child_benefit_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "child_benefit_child_index": -1,
                    "child_benefit_respective_amount": 0,
                },
                {
                    "person_id": 8,
                    "child_benefit_child_index": 1,
                    "child_benefit_respective_amount": 27.05 * WEEKS_IN_YEAR,
                },
            ],
            "person_ids": [7, 8],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_8",
            "period": {
                "period_kind": "custom",
                "name": "benefit_week",
                "start": "2026-04-06",
                "end": "2026-04-12",
            },
            "outputs": [CHILD_BENEFIT_OUTPUTS["child_benefit_weekly_rate"]["axiom"]],
        }
    ]
    assert {record["name"]: record["value"] for record in request["dataset"]["inputs"]}[
        f"{CHILD_BENEFIT_BASE}#input.child_or_qualifying_young_person_is_only_elder_or_eldest_for_payee"
    ] == {
        "kind": "bool",
        "value": True,
    }


def test_select_person_indices_uses_positive_weights_and_explicit_ids():
    rows = [
        {"person_id": 10, "person_weight": 0},
        {"person_id": 11, "person_weight": 2.5},
        {"person_id": 12, "person_weight": 3.5},
    ]

    assert select_person_indices(rows, sample_size=1) == [1]
    assert select_person_indices(rows, sample_size=0) == [1, 2]
    assert select_person_indices(rows, sample_size=1, person_ids=(12,)) == [2]

    with pytest.raises(SystemExit, match="not eligible"):
        select_person_indices(rows, sample_size=0, person_ids=(10,))


def test_policyengine_variables_for_surfaces_deduplicates_person_variables():
    assert policyengine_person_variables_for_surfaces(
        ["personal-allowance", "child-benefit"]
    ) == (
        "adjusted_net_income",
        "child_benefit_child_index",
        "child_benefit_respective_amount",
        "gift_aid_grossed_up",
        "personal_allowance",
    )


def test_compare_outputs_reports_personal_allowance_mismatch():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "personal_allowance": 12_570,
                }
            ],
            "person_ids": [7],
        },
        axiom_outputs_by_surface={
            "personal-allowance": [
                {
                    "outputs": {
                        PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"][
                            "axiom"
                        ]: decimal_output(12_569)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_persons == 1
    assert report.compared_values == 1
    assert len(report.mismatches) == 1
    assert report.oracle_divergences == []
    assert report.mismatches[0].entity_id == "person_7"
    assert report.output_summary[0]["mismatches"] == 1
    assert {item["surface"] for item in report.skipped_surfaces} == {
        "pension-credit",
        "universal-credit",
    }


def test_compare_outputs_classifies_known_policyengine_personal_allowance_rounding():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "personal_allowance": 12_569.5,
                }
            ],
            "person_ids": [7],
        },
        axiom_outputs_by_surface={
            "personal-allowance": [
                {
                    "outputs": {
                        PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"][
                            "axiom"
                        ]: decimal_output(12_570)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1738")
    assert report.output_summary[0]["oracle_divergences"] == 1


def test_compare_outputs_compares_child_benefit_weekly_rate():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "child_benefit_child_index": -1,
                    "child_benefit_respective_amount": 0,
                },
                {
                    "person_id": 8,
                    "child_benefit_child_index": 1,
                    "child_benefit_respective_amount": 27.05 * WEEKS_IN_YEAR,
                },
            ],
            "person_ids": [7, 8],
        },
        axiom_outputs_by_surface={
            "child-benefit": [
                {
                    "outputs": {
                        CHILD_BENEFIT_OUTPUTS["child_benefit_weekly_rate"][
                            "axiom"
                        ]: decimal_output(27.05)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_persons == 2
    assert report.compared_values == 1
    assert report.mismatches == []
    assert report.oracle_divergences == []
    assert report.output_summary[0]["compared"] == 1


def test_compare_outputs_classifies_known_policyengine_child_benefit_amounts():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 8,
                    "child_benefit_child_index": 1,
                    "child_benefit_respective_amount": 26.935709699992934
                    * WEEKS_IN_YEAR,
                },
            ],
            "person_ids": [8],
        },
        axiom_outputs_by_surface={
            "child-benefit": [
                {
                    "outputs": {
                        CHILD_BENEFIT_OUTPUTS["child_benefit_weekly_rate"][
                            "axiom"
                        ]: decimal_output(27.05)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1739")


def test_compare_uk_efrs_runs_axiom_personal_allowance(
    monkeypatch,
    tmp_path,
):
    rulespec_root = tmp_path / "rulespec-uk"
    program = rulespec_root / PERSONAL_ALLOWANCE_PROGRAM_PATH
    program.parent.mkdir(parents=True)
    program.write_text("format: rulespec/v1\n")
    axiom_rules = tmp_path / "axiom-rules-engine"
    axiom_rules.mkdir()
    captured = {}

    monkeypatch.setattr(
        efrs_uk,
        "load_policyengine_uk_data",
        lambda **_: {
            "persons": [
                {
                    "person_id": 7,
                    "person_weight": 1,
                    "adjusted_net_income": 20_000,
                    "gift_aid_grossed_up": 0,
                    "personal_allowance": 12_570,
                }
            ],
            "person_ids": [7],
        },
    )

    def fake_run_axiom_program(**kwargs):
        captured.update(kwargs)
        return [
            {
                "outputs": {
                    PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"][
                        "axiom"
                    ]: decimal_output(12_570)
                }
            }
        ]

    monkeypatch.setattr(efrs_uk, "run_axiom_program", fake_run_axiom_program)

    report = compare_uk_efrs(
        workspace_root=tmp_path,
        rulespec_root=None,
        axiom_rules_path=None,
        year=2026,
        sample_size=100,
        surface="personal-allowance",
        dataset="enhanced_frs_2023_24",
        data_folder=Path(".axiom/policyengine-data"),
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []
    assert captured["program"] == program
    assert captured["rulespec_root"] == rulespec_root.resolve()
    assert captured["axiom_rules_path"] == axiom_rules.resolve()
    assert captured["request"]["queries"][0]["entity_id"] == "person_7"


def test_main_returns_nonzero_when_requested_for_mismatches(monkeypatch, tmp_path):
    monkeypatch.setattr(efrs_uk, "resolve_workspace_root", lambda root: tmp_path)
    monkeypatch.setattr(
        efrs_uk,
        "compare_uk_efrs",
        lambda **_: efrs_uk.UKEFRSComparisonReport(
            compared_persons=1,
            compared_values=1,
            mismatches=[
                efrs_uk.UKEFRSComparisonRow(
                    surface="personal-allowance",
                    entity_id="person_7",
                    output="personal_allowance",
                    axiom=0,
                    policyengine=1,
                    diff=-1,
                )
            ],
            oracle_divergences=[],
            output_summary=[],
            skipped_surfaces=[],
            projection_notes=[],
        ),
    )

    assert (
        efrs_uk.main(
            argparse.Namespace(
                root=None,
                rulespec_root=None,
                axiom_rules_engine_path=None,
                year=2026,
                sample_size=100,
                surface="all",
                dataset="enhanced_frs_2023_24",
                data_folder=Path(".axiom/policyengine-data"),
                tolerance=0.01,
                relative_tolerance=2e-7,
                person_ids=None,
                json=True,
                fail_on_mismatch=True,
            )
        )
        == 1
    )

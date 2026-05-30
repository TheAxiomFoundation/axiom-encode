import argparse
from pathlib import Path

import pytest

import axiom_encode.oracles.policyengine.efrs_uk as efrs_uk
from axiom_encode.oracles.policyengine.efrs_uk import (
    PERSONAL_ALLOWANCE_BASE,
    PERSONAL_ALLOWANCE_OUTPUTS,
    PERSONAL_ALLOWANCE_PROGRAM_PATH,
    build_personal_allowance_request,
    compare_outputs,
    compare_uk_efrs,
    policyengine_person_variables_for_surfaces,
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
    assert policyengine_person_variables_for_surfaces(["personal-allowance"]) == (
        "adjusted_net_income",
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
    assert report.mismatches[0].entity_id == "person_7"
    assert report.output_summary[0]["mismatches"] == 1
    assert {item["surface"] for item in report.skipped_surfaces} == {
        "child-benefit",
        "pension-credit",
        "universal-credit",
    }


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
        surface="all",
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

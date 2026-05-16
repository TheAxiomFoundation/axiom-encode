import pytest

from axiom_encode.oracles.policyengine.ecps_tax import (
    OASDI_WAGE_BASE_BASE,
    OASDI_WAGE_BASE_EXCLUSION_OUTPUT,
    additional_standard_deduction_entitlement_count,
    build_oasdi_wage_base_request,
    build_payroll_request,
    filing_status_code,
    individual_is_unmarried_and_not_surviving_spouse,
    person_entity_id,
    project_ctc_h_person_inputs,
    project_ctc_person_inputs,
    project_oasdi_wage_base_inputs,
    project_standard_deduction_inputs,
    project_tax_unit_inputs,
    project_tax_unit_person_contexts,
    social_security_contribution_and_benefit_base,
    taxable_oasdi_wages_by_person_id,
    uses_joint_ctc_phaseout_threshold,
    valid_child_ssn_type,
    within_tolerance,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("SINGLE", 0),
        ("JOINT", 1),
        ("SEPARATE", 2),
        ("HEAD_OF_HOUSEHOLD", 3),
        ("SURVIVING_SPOUSE", 4),
    ],
)
def test_filing_status_code_maps_policyengine_enum(value, expected):
    assert filing_status_code(value) == expected


def test_person_entity_id_is_stable_and_namespaced():
    assert person_entity_id(42) == "person_42"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("CITIZEN", True),
        ("NON_CITIZEN_VALID_EAD", True),
        ("NONE", False),
        ("OTHER_NON_CITIZEN", False),
    ],
)
def test_valid_child_ssn_type_maps_policyengine_enum(value, expected):
    assert valid_child_ssn_type(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("JOINT", True),
        ("SURVIVING_SPOUSE", True),
        ("SINGLE", False),
        ("SEPARATE", False),
        ("HEAD_OF_HOUSEHOLD", False),
    ],
)
def test_uses_joint_ctc_phaseout_threshold_matches_policyengine(value, expected):
    assert uses_joint_ctc_phaseout_threshold(value) is expected


def test_ctc_person_projection_marks_under_17_valid_ssn_child():
    projected = project_ctc_person_inputs({"age": 16, "ssn_card_type": "CITIZEN"})

    assert projected["age"] == 16
    assert projected["qualifying_child_under_section_152_c"] is True
    assert projected["allowed_deduction_under_section_151_for_child"] is True
    assert projected["ctc_child_missing_identification"] is False
    assert projected["qualifying_child_tin_included_on_return"] is True


def test_ctc_person_projection_marks_under_17_missing_ssn_child():
    projected = project_ctc_person_inputs({"age": 9, "ssn_card_type": "NONE"})
    subsection_h_projected = project_ctc_h_person_inputs(
        {"age": 9, "ssn_card_type": "NONE"}
    )

    assert projected["qualifying_child_under_section_152_c"] is True
    assert projected["ctc_child_missing_identification"] is True
    assert projected["qualifying_child_tin_included_on_return"] is False
    assert subsection_h_projected["dependent_under_section_152"] is True
    assert (
        subsection_h_projected["qualifying_child_ssn_is_valid_for_subsection_h"]
        is False
    )


def test_ctc_person_projection_treats_single_adult_as_tax_unit_head():
    projected = project_ctc_person_inputs({"age": 18, "ssn_card_type": "CITIZEN"})
    subsection_h_projected = project_ctc_h_person_inputs(
        {"age": 18, "ssn_card_type": "CITIZEN"}
    )

    assert projected["qualifying_child_under_section_152_c"] is False
    assert projected["allowed_deduction_under_section_151_for_child"] is False
    assert subsection_h_projected["dependent_under_section_152"] is False


def test_tax_unit_person_contexts_project_head_spouse_and_dependents():
    contexts = project_tax_unit_person_contexts(
        [
            {"age": 42, "ssn_card_type": "CITIZEN"},
            {"age": 58, "ssn_card_type": "CITIZEN"},
            {"age": 24, "ssn_card_type": "CITIZEN"},
            {"age": 18, "ssn_card_type": "CITIZEN"},
            {"age": 15, "ssn_card_type": "CITIZEN"},
        ]
    )

    assert [context.is_head for context in contexts] == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert [context.is_spouse for context in contexts] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert [context.is_tax_unit_dependent for context in contexts] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert [
        context.qualifying_child_described_in_subsection_c for context in contexts
    ] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert all(context.filer_has_valid_child_ctc_ssn for context in contexts)


def test_tax_unit_person_contexts_handle_minor_only_tax_unit_without_filer_ssn():
    [context] = project_tax_unit_person_contexts(
        [{"age": 16, "ssn_card_type": "CITIZEN"}]
    )
    projected = project_ctc_h_person_inputs(
        {"age": 16, "ssn_card_type": "CITIZEN"}, context
    )

    assert context.is_tax_unit_dependent is True
    assert context.qualifying_child_described_in_subsection_c is True
    assert context.filer_has_valid_child_ctc_ssn is False
    assert projected["taxpayer_or_spouse_ssn_is_valid_for_subsection_h"] is False


def test_tax_unit_projection_uses_boundary_inputs_without_ctc_outputs():
    projected = project_tax_unit_inputs(
        {"adjusted_gross_income": 123_456, "filing_status": "HEAD_OF_HOUSEHOLD"}
    )

    assert projected["adjusted_gross_income"] == 123_456
    assert projected["filing_status"] == 3
    assert projected["aggregate_advance_payments_under_section_7527A"] == 0
    assert "ctc" not in projected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("SINGLE", True),
        ("HEAD_OF_HOUSEHOLD", True),
        ("JOINT", False),
        ("SEPARATE", False),
        ("SURVIVING_SPOUSE", False),
    ],
)
def test_unmarried_not_surviving_spouse_matches_standard_deduction_status(
    value, expected
):
    assert individual_is_unmarried_and_not_surviving_spouse(value) is expected


def test_additional_standard_deduction_count_uses_head_and_spouse_only():
    count = additional_standard_deduction_entitlement_count(
        [
            {"age": 68, "is_blind": True},
            {"age": 70, "is_blind": False},
            {"age": 80, "is_blind": True},
        ]
    )

    assert count == 3


def test_standard_deduction_projection_uses_leaf_age_and_blind_inputs():
    projected = project_standard_deduction_inputs(
        row={"filing_status": "HEAD_OF_HOUSEHOLD"},
        persons=[
            {"age": 66, "is_blind": True},
            {"age": 15, "is_blind": True},
        ],
    )

    assert projected["filing_status"] == 3
    assert projected["may_be_claimed_as_dependent_by_another_taxpayer"] is False
    assert projected["earned_income"] == 0
    assert (
        projected["additional_standard_deduction_entitlement_count_under_subsection_f"]
        == 2
    )
    assert projected["individual_is_unmarried_and_not_surviving_spouse"] is True


def test_within_tolerance_keeps_cent_level_strictness_for_ordinary_outputs():
    assert within_tolerance(
        1000.005,
        1000,
        absolute_tolerance=0.01,
        relative_tolerance=2e-7,
    )
    assert not within_tolerance(
        1000.02,
        1000,
        absolute_tolerance=0.01,
        relative_tolerance=2e-7,
    )


def test_within_tolerance_accepts_large_policyengine_float_noise():
    assert within_tolerance(
        1_271_556_450,
        1_271_556_352,
        absolute_tolerance=0.01,
        relative_tolerance=2e-7,
    )


def test_oasdi_wage_base_projection_uses_gross_wages_and_official_base():
    projected = project_oasdi_wage_base_inputs(
        {"payroll_tax_gross_wages": 200_000},
        contribution_base=social_security_contribution_and_benefit_base(2026),
    )

    assert (
        projected[
            "contribution_and_benefit_base_under_section_230_of_social_security_act"
        ]
        == 184_500
    )
    assert (
        projected[
            "remuneration_paid_to_individual_by_employer_with_respect_to_employment"
        ]
        == 200_000
    )
    assert (
        projected[
            "remuneration_other_than_succeeding_paragraphs_paid_to_individual_by_employer_during_calendar_year_before_payment"
        ]
        == 0
    )


def test_build_oasdi_wage_base_request_uses_3121_inputs():
    pd = pytest.importorskip("pandas")
    pe_data = {
        "persons": pd.DataFrame(
            [{"person_id": 7, "payroll_tax_gross_wages": 200_000}]
        ),
        "person_ids": [7],
    }

    request = build_oasdi_wage_base_request(pe_data=pe_data, year=2026)

    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": [OASDI_WAGE_BASE_EXCLUSION_OUTPUT],
        }
    ]
    input_values = {
        item["name"]: item["value"]["value"] for item in request["dataset"]["inputs"]
    }
    assert (
        input_values[
            f"{OASDI_WAGE_BASE_BASE}#input.contribution_and_benefit_base_under_section_230_of_social_security_act"
        ]
        == "184500.0"
    )
    assert (
        input_values[
            f"{OASDI_WAGE_BASE_BASE}#input.remuneration_paid_to_individual_by_employer_with_respect_to_employment"
        ]
        == "200000.0"
    )


def test_taxable_oasdi_wages_come_from_axiom_3121_results():
    pd = pytest.importorskip("pandas")
    pe_data = {
        "persons": pd.DataFrame(
            [{"person_id": 7, "payroll_tax_gross_wages": 200_000}]
        ),
        "person_ids": [7],
    }
    results = [
        {
            "outputs": {
                OASDI_WAGE_BASE_EXCLUSION_OUTPUT: {
                    "kind": "scalar",
                    "value": {"kind": "decimal", "value": "15500"},
                }
            }
        }
    ]

    assert taxable_oasdi_wages_by_person_id(
        pe_data=pe_data,
        oasdi_wage_base_results=results,
    ) == {7: 184_500}


def test_build_oasdi_payroll_request_feeds_3121_taxable_wages():
    pd = pytest.importorskip("pandas")
    pe_data = {
        "persons": pd.DataFrame(
            [{"person_id": 7, "payroll_tax_gross_wages": 200_000}]
        ),
        "person_ids": [7],
    }
    results = [
        {
            "outputs": {
                OASDI_WAGE_BASE_EXCLUSION_OUTPUT: {
                    "kind": "scalar",
                    "value": {"kind": "decimal", "value": "15500"},
                }
            }
        }
    ]

    request = build_payroll_request(
        pe_data=pe_data,
        year=2026,
        surface="employee-oasdi",
        oasdi_wage_base_results=results,
    )

    assert request["dataset"]["inputs"] == [
        {
            "name": "us:statutes/26/3101/a#input.wages",
            "entity": "Entity",
            "entity_id": "person_7",
            "interval": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "value": {"kind": "decimal", "value": "184500.0"},
        }
    ]


def test_build_oasdi_payroll_request_requires_3121_results():
    pd = pytest.importorskip("pandas")
    pe_data = {
        "persons": pd.DataFrame(
            [{"person_id": 7, "payroll_tax_gross_wages": 200_000}]
        ),
        "person_ids": [7],
    }

    with pytest.raises(ValueError, match="Axiom 3121 wage-base results"):
        build_payroll_request(
            pe_data=pe_data,
            year=2026,
            surface="employee-oasdi",
        )

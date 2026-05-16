import pytest

from axiom_encode.oracles.policyengine.ecps_tax import (
    filing_status_code,
    project_ctc_h_person_inputs,
    project_ctc_person_inputs,
    project_tax_unit_inputs,
    project_tax_unit_person_contexts,
    uses_joint_ctc_phaseout_threshold,
    valid_child_ssn_type,
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

from pathlib import Path

import pytest

from axiom_encode.oracles.policyengine import medicaid_populace


class FakeUSDataset:
    def __init__(
        self,
        *,
        person,
        household,
        tax_unit,
        spm_unit=None,
        family=None,
        marital_unit=None,
        time_period=2024,
    ):
        self.person = person
        self.household = household
        self.tax_unit = tax_unit
        self.spm_unit = spm_unit
        self.family = family
        self.marital_unit = marital_unit
        self.time_period = str(time_period)


def test_pre_sample_households_preserves_whole_state_households():
    pd = pytest.importorskip("pandas")
    dataset = FakeUSDataset(
        household=pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "state_fips": [36, 6, 6],
            }
        ),
        person=pd.DataFrame(
            {
                "person_id": [10, 20, 21, 30],
                "person_household_id": [1, 2, 2, 3],
                "person_tax_unit_id": [100, 200, 200, 300],
                "person_spm_unit_id": [1000, 2000, 2000, 3000],
                "person_family_id": [10000, 20000, 20000, 30000],
                "person_marital_unit_id": [100000, 200000, 200001, 300000],
            }
        ),
        tax_unit=pd.DataFrame({"tax_unit_id": [100, 200, 300]}),
        spm_unit=pd.DataFrame({"spm_unit_id": [1000, 2000, 3000]}),
        family=pd.DataFrame({"family_id": [10000, 20000, 30000]}),
        marital_unit=pd.DataFrame(
            {"marital_unit_id": [100000, 200000, 200001, 300000]}
        ),
    )

    sampled = medicaid_populace.pre_sample_households(
        dataset,
        household_count=1,
        state_filter="CA",
    )

    assert sampled.household["household_id"].tolist() == [2]
    assert sampled.person["person_id"].tolist() == [20, 21]
    assert sampled.tax_unit["tax_unit_id"].tolist() == [200]
    assert sampled.spm_unit["spm_unit_id"].tolist() == [2000]
    assert sampled.family["family_id"].tolist() == [20000]
    assert sampled.marital_unit["marital_unit_id"].tolist() == [200000, 200001]


def test_runtime_output_id_for_public_id_resolves_branch_qualified_id(tmp_path: Path):
    artifact = tmp_path / "program.json"
    artifact.write_text(
        """
        {
          "program": {
            "derived": [
              {
                "id": "us-medicaid-branch:us/statutes/42/1396a/a/10#is_medicaid_eligible",
                "name": "is_medicaid_eligible"
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    assert (
        medicaid_populace.runtime_output_id_for_public_id(
            artifact,
            "us:statutes/42/1396a/a/10#is_medicaid_eligible",
        )
        == "us-medicaid-branch:us/statutes/42/1396a/a/10#is_medicaid_eligible"
    )


def test_project_case_inputs_uses_shared_income_projection_for_medicaid_imports():
    inputs = medicaid_populace._project_case_inputs(
        {},
        age=10,
        medicaid_income_level=0.25,
        parent_nfc=False,
        parent_fc=False,
        pregnant_nfc=False,
        pregnant_fc=False,
        infant_fc=False,
        young_child_fc=False,
        older_child_eligible=True,
        adult_nfc=False,
        adult_fc=True,
        ssi_recipient=False,
        mandatory_subpart_b=True,
        work_requirement_eligible=True,
        medicare_eligible=False,
    )

    assert (
        inputs[
            "us:regulations/42-cfr/435/116#input.household_income_as_fraction_of_fpl"
        ]
        == 1.0
    )
    assert (
        inputs[
            "us:regulations/42-cfr/435/118#input.household_income_as_fraction_of_fpl"
        ]
        == 1.0
    )
    assert (
        inputs[
            "us:regulations/42-cfr/435/119#input.household_income_as_fraction_of_fpl"
        ]
        == 1.0
    )


def test_summarize_by_pe_category_counts_directional_mismatches():
    rows = [
        {
            "pe_medicaid_category": "SENIOR_OR_DISABLED",
            "pe_is_medicaid_eligible": True,
            "axiom_is_medicaid_eligible": False,
            "match": False,
        },
        {
            "pe_medicaid_category": "SENIOR_OR_DISABLED",
            "pe_is_medicaid_eligible": True,
            "axiom_is_medicaid_eligible": True,
            "match": True,
        },
        {
            "pe_medicaid_category": "ADULT",
            "pe_is_medicaid_eligible": False,
            "axiom_is_medicaid_eligible": True,
            "match": False,
        },
    ]

    summary = medicaid_populace.summarize_by_pe_category(rows)

    assert summary == [
        {
            "category": "SENIOR_OR_DISABLED",
            "compared": 2,
            "pe_eligible": 2,
            "axiom_eligible": 1,
            "mismatches": 1,
            "pe_true_axiom_false": 1,
            "pe_false_axiom_true": 0,
        },
        {
            "category": "ADULT",
            "compared": 1,
            "pe_eligible": 0,
            "axiom_eligible": 1,
            "mismatches": 1,
            "pe_true_axiom_false": 0,
            "pe_false_axiom_true": 1,
        },
    ]

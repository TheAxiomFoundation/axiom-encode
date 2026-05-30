"""Compare UK RuleSpec output against PolicyEngine Enhanced FRS.

The UK counterpart to the ECPS tax comparators covers mapped surfaces whose
RuleSpec inputs can be projected from PolicyEngine's Enhanced FRS, plus scalar
parameter surfaces whose generated RuleSpec values can be compared against
PolicyEngine component outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .ecps_tax import (
    POLICYENGINE_VERSION,
    input_record,
    money,
    output_number,
    relative_diff,
    run_axiom_program,
    within_tolerance,
)

DEFAULT_DATASET = "enhanced_frs_2023_24"
WEEKS_IN_YEAR = 52
MONTHS_IN_YEAR = 12

PERSONAL_ALLOWANCE_PROGRAM_PATH = Path("statutes/ukpga/2007/3/35.yaml")
PERSONAL_ALLOWANCE_BASE = "uk:statutes/ukpga/2007/3/35"
CHILD_BENEFIT_PROGRAM_PATH = Path("regulations/uksi/2006/965/2.yaml")
CHILD_BENEFIT_BASE = "uk:regulations/uksi/2006/965/2"
PENSION_CREDIT_PROGRAM_PATH = Path("regulations/uksi/2002/1792/6.yaml")
PENSION_CREDIT_BASE = "uk:regulations/uksi/2002/1792/6"
UNIVERSAL_CREDIT_PROGRAM_PATH = Path("regulations/uksi/2013/376/36.yaml")
UNIVERSAL_CREDIT_BASE = "uk:regulations/uksi/2013/376/36"

PERSONAL_ALLOWANCE_OUTPUTS = {
    "personal_allowance": {
        "axiom": f"{PERSONAL_ALLOWANCE_BASE}#personal_allowance",
        "pe": "personal_allowance",
    },
}

CHILD_BENEFIT_OUTPUTS = {
    "child_benefit_weekly_rate": {
        "axiom": f"{CHILD_BENEFIT_BASE}#child_benefit_weekly_rate",
        "pe": "child_benefit_respective_amount",
        "pe_transform": "annual_to_weekly",
    },
}

PENSION_CREDIT_OUTPUTS = {
    "standard_minimum_guarantee": {
        "axiom": f"{PENSION_CREDIT_BASE}#standard_minimum_guarantee",
        "pe": "standard_minimum_guarantee",
        "pe_transform": "annual_to_weekly",
    },
    "severe_disability_additional_amount": {
        "axiom": f"{PENSION_CREDIT_BASE}#severe_disability_additional_amount",
        "pe": "severe_disability_minimum_guarantee_addition",
        "pe_transform": "annual_to_weekly",
    },
    "carer_additional_amount": {
        "axiom": f"{PENSION_CREDIT_BASE}#carer_additional_amount",
        "pe": "carer_minimum_guarantee_addition",
        "pe_transform": "annual_to_weekly_per_carer",
    },
}

UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS = {
    "standard_allowance_single_under_25": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_single_under_25",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "SINGLE_YOUNG"),
    },
    "standard_allowance_single_25_or_over": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_single_25_or_over",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "SINGLE_OLD"),
    },
    "standard_allowance_joint_both_under_25": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_joint_both_under_25",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "COUPLE_YOUNG"),
    },
    "standard_allowance_joint_either_25_or_over": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_joint_either_25_or_over",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "COUPLE_OLD"),
    },
}

UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS = {
    "child_element_first_child_or_qualifying_young_person": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#child_element_first_child_or_qualifying_young_person",
        "pe": "uc_individual_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_first_child_element",
    },
    "child_element_second_and_each_subsequent_child_or_qualifying_young_person": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#child_element_second_and_each_subsequent_child_or_qualifying_young_person",
        "pe": "uc_individual_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_subsequent_child_element",
    },
    "disabled_child_additional_amount_lower_rate": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#disabled_child_additional_amount_lower_rate",
        "pe": "uc_individual_disabled_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "positive_pe_output",
    },
    "disabled_child_additional_amount_higher_rate": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#disabled_child_additional_amount_higher_rate",
        "pe": "uc_individual_severely_disabled_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "positive_pe_output",
    },
}

UNIVERSAL_CREDIT_LCWRA_OUTPUTS = {
    "lcwra_element_standard_lcwra_claimant": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#lcwra_element_standard_lcwra_claimant",
        "pe": "uc_LCWRA_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_lcwra_standard_amount",
    },
    "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant",
        "pe": "uc_LCWRA_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_lcwra_higher_amount",
    },
}

UNIVERSAL_CREDIT_CARER_OUTPUTS = {
    "carer_element": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#carer_element",
        "pe": "uc_carer_element",
        "pe_transform": "annual_to_monthly",
        "applies": "positive_pe_output",
    },
}

UNIVERSAL_CREDIT_CHILDCARE_OUTPUTS = {
    "childcare_costs_element_maximum_one_child": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#childcare_costs_element_maximum_one_child",
        "pe": "uc_maximum_childcare_element_amount",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_childcare_element_eligible_children", 1),
    },
    "childcare_costs_element_maximum_two_or_more_children": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#childcare_costs_element_maximum_two_or_more_children",
        "pe": "uc_maximum_childcare_element_amount",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_childcare_two_or_more_children",
    },
}

UNIVERSAL_CREDIT_2026_RULESPEC_RATES = {
    "standard_allowance_single_under_25": 338.58,
    "standard_allowance_single_25_or_over": 424.90,
    "standard_allowance_joint_both_under_25": 528.34,
    "standard_allowance_joint_either_25_or_over": 666.97,
    "child_element_first_child_or_qualifying_young_person": 351.88,
    "child_element_second_and_each_subsequent_child_or_qualifying_young_person": 303.94,
    "disabled_child_additional_amount_lower_rate": 164.79,
    "disabled_child_additional_amount_higher_rate": 514.71,
    "lcwra_element_standard_lcwra_claimant": 217.26,
    "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant": 429.80,
    "carer_element": 209.34,
    "childcare_costs_element_maximum_one_child": 1071.09,
    "childcare_costs_element_maximum_two_or_more_children": 1836.16,
}


@dataclass(frozen=True)
class UKEFRSSurfaceSpec:
    program: Path
    entity: str
    outputs: dict[str, dict[str, Any]]
    pe_variables: tuple[str, ...]


SURFACE_SPECS = {
    "personal-allowance": UKEFRSSurfaceSpec(
        program=PERSONAL_ALLOWANCE_PROGRAM_PATH,
        entity="person",
        outputs=PERSONAL_ALLOWANCE_OUTPUTS,
        pe_variables=(
            "adjusted_net_income",
            "gift_aid_grossed_up",
            "personal_allowance",
        ),
    ),
    "child-benefit": UKEFRSSurfaceSpec(
        program=CHILD_BENEFIT_PROGRAM_PATH,
        entity="person",
        outputs=CHILD_BENEFIT_OUTPUTS,
        pe_variables=(
            "child_benefit_child_index",
            "child_benefit_respective_amount",
        ),
    ),
    "pension-credit": UKEFRSSurfaceSpec(
        program=PENSION_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=PENSION_CREDIT_OUTPUTS,
        pe_variables=(
            "carer_minimum_guarantee_addition",
            "is_couple",
            "num_carers",
            "relation_type",
            "severe_disability_minimum_guarantee_addition",
            "standard_minimum_guarantee",
        ),
    ),
    "universal-credit-standard-allowance": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS,
        pe_variables=(
            "uc_standard_allowance",
            "uc_standard_allowance_claimant_type",
        ),
    ),
    "universal-credit-child-element": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="person",
        outputs=UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS,
        pe_variables=(
            "uc_child_index",
            "uc_individual_child_element",
            "uc_individual_disabled_child_element",
            "uc_individual_severely_disabled_child_element",
            "uc_is_child_born_before_child_limit",
        ),
    ),
    "universal-credit-lcwra-element": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_LCWRA_OUTPUTS,
        pe_variables=("uc_LCWRA_element",),
    ),
    "universal-credit-carer-element": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_CARER_OUTPUTS,
        pe_variables=("uc_carer_element",),
    ),
    "universal-credit-childcare-cap": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_CHILDCARE_OUTPUTS,
        pe_variables=(
            "uc_childcare_element_eligible_children",
            "uc_maximum_childcare_element_amount",
        ),
    ),
}

SKIPPED_SURFACES: list[dict[str, str]] = []


@dataclass(frozen=True)
class UKEFRSComparisonRow:
    surface: str
    entity_id: str
    output: str
    axiom: float
    policyengine: float
    diff: float


@dataclass(frozen=True)
class UKEFRSOracleDivergence(UKEFRSComparisonRow):
    reason: str
    issue_url: str


@dataclass(frozen=True)
class UKEFRSComparisonReport:
    compared_persons: int
    compared_benunits: int
    compared_values: int
    mismatches: list[UKEFRSComparisonRow]
    oracle_divergences: list[UKEFRSOracleDivergence]
    output_summary: list[dict[str, Any]]
    skipped_surfaces: list[dict[str, str]]
    projection_notes: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "compared_persons": self.compared_persons,
            "compared_benunits": self.compared_benunits,
            "compared_values": self.compared_values,
            "mismatch_count": len(self.mismatches),
            "mismatches": [row.__dict__ for row in self.mismatches],
            "oracle_divergence_count": len(self.oracle_divergences),
            "oracle_divergences": [row.__dict__ for row in self.oracle_divergences],
            "output_summary": self.output_summary,
            "skipped_surfaces": self.skipped_surfaces,
            "projection_notes": self.projection_notes,
        }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root containing rulespec-uk and axiom-rules-engine",
    )
    parser.add_argument(
        "--rulespec-root",
        type=Path,
        default=None,
        help="rulespec-uk checkout; defaults to <root>/rulespec-uk",
    )
    parser.add_argument(
        "--axiom-rules-engine-path",
        type=Path,
        default=None,
        help="axiom-rules-engine checkout; defaults to <root>/axiom-rules-engine",
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help=(
            "Number of positive-weight EFRS people to compare; "
            "0 compares all eligible people"
        ),
    )
    parser.add_argument(
        "--person-id",
        type=int,
        action="append",
        default=None,
        dest="person_ids",
        help=(
            "Compare a specific EFRS person_id. Repeat to compare multiple "
            "known residual cases; when provided this bypasses --sample-size."
        ),
    )
    parser.add_argument(
        "--surface",
        choices=["all", *SURFACE_SPECS],
        default="all",
        help="UK EFRS surface to compare; defaults to all implemented surfaces",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "PolicyEngine UK dataset logical name, HuggingFace URI, or local .h5 "
            "path. Defaults to enhanced_frs_2023_24 and prefers local managed "
            "mirrors when available."
        ),
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path(".axiom") / "policyengine-data",
        help="PolicyEngine dataset cache folder",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Absolute tolerance for matching PolicyEngine outputs",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=2e-7,
        help=(
            "Relative tolerance for large floating PolicyEngine intermediates; "
            "ordinary pound outputs remain controlled by --tolerance"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero when any compared value differs beyond tolerance",
    )


def main(args: argparse.Namespace) -> int:
    report = compare_uk_efrs(
        workspace_root=resolve_workspace_root(args.root),
        rulespec_root=args.rulespec_root,
        axiom_rules_path=args.axiom_rules_engine_path,
        year=args.year,
        sample_size=args.sample_size,
        surface=args.surface,
        dataset=args.dataset,
        data_folder=args.data_folder,
        tolerance=args.tolerance,
        relative_tolerance=args.relative_tolerance,
        person_ids=tuple(args.person_ids or ()),
    )
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        print_report(
            report,
            tolerance=args.tolerance,
            relative_tolerance=args.relative_tolerance,
        )
    if args.fail_on_mismatch and report.mismatches:
        return 1
    return 0


def compare_uk_efrs(
    *,
    workspace_root: Path,
    rulespec_root: Path | None,
    axiom_rules_path: Path | None,
    year: int,
    sample_size: int,
    surface: str,
    dataset: str,
    data_folder: Path,
    tolerance: float,
    relative_tolerance: float,
    person_ids: tuple[int, ...] = (),
) -> UKEFRSComparisonReport:
    resolved_rulespec_root = (rulespec_root or workspace_root / "rulespec-uk").resolve()
    resolved_axiom_rules_path = (
        axiom_rules_path or workspace_root / "axiom-rules-engine"
    ).resolve()
    surfaces = list(SURFACE_SPECS) if surface == "all" else [surface]
    pe_data = load_policyengine_uk_data(
        year=year,
        sample_size=sample_size,
        dataset=dataset,
        data_folder=data_folder,
        person_ids=person_ids,
        person_variables=policyengine_person_variables_for_surfaces(surfaces),
        benunit_variables=policyengine_benunit_variables_for_surfaces(surfaces),
    )
    surface_results: dict[str, list[dict[str, Any]]] = {}
    for selected_surface in surfaces:
        spec = SURFACE_SPECS[selected_surface]
        program = resolved_rulespec_root / spec.program
        if not program.exists():
            raise SystemExit(f"{selected_surface} RuleSpec not found: {program}")
        request = build_axiom_request(
            pe_data=pe_data,
            year=year,
            surface=selected_surface,
        )
        surface_results[selected_surface] = run_axiom_surface(
            program=program,
            request=request,
            rulespec_root=resolved_rulespec_root,
            axiom_rules_path=resolved_axiom_rules_path,
            surface=selected_surface,
        )
    return compare_outputs(
        pe_data=pe_data,
        axiom_outputs_by_surface=surface_results,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )


def load_policyengine_uk_data(
    *,
    year: int,
    sample_size: int,
    dataset: str,
    data_folder: Path,
    person_ids: tuple[int, ...] = (),
    person_variables: tuple[str, ...] = SURFACE_SPECS[
        "personal-allowance"
    ].pe_variables,
    benunit_variables: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        from policyengine.core import Simulation
        from policyengine.provenance.manifest import dataset_logical_name
        from policyengine.tax_benefit_models.uk import ensure_datasets, uk_latest
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc

    log("Loading PolicyEngine UK EFRS...")
    local_dataset = local_policyengine_uk_dataset(
        dataset=dataset,
        year=year,
    )
    if local_dataset is not None:
        pe_dataset = local_dataset
    else:
        resolved_dataset = resolve_policyengine_uk_dataset_reference(dataset)
        datasets = ensure_datasets(
            datasets=[resolved_dataset],
            years=[year],
            data_folder=str(data_folder),
        )
        pe_dataset = datasets[f"{dataset_logical_name(resolved_dataset)}_{year}"]
    extra_variables: dict[str, list[str]] = {}
    if person_variables:
        extra_variables["person"] = list(person_variables)
    if benunit_variables:
        extra_variables["benunit"] = list(benunit_variables)
    sim = Simulation(
        dataset=pe_dataset,
        tax_benefit_model_version=uk_latest,
        extra_variables=extra_variables,
    )
    log("Running PolicyEngine UK outputs...")
    sim.run()

    person_columns = ["person_id", "person_weight"]
    if "person_benunit_id" in pe_dataset.data.person.columns:
        person_columns.append("person_benunit_id")
    raw_persons = pe_dataset.data.person[person_columns].copy()
    person_outputs = sim.output_dataset.data.person[["person_id", *person_variables]]
    merged = raw_persons.merge(
        person_outputs,
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    records = table_records(merged)
    selected_indices = select_person_indices(
        records,
        sample_size=sample_size,
        person_ids=person_ids,
    )
    selected = [records[index] for index in selected_indices]
    benunit_records: list[dict[str, Any]] = []
    selected_benunits: list[dict[str, Any]] = []
    if benunit_variables:
        raw_benunits = pe_dataset.data.benunit[["benunit_id", "benunit_weight"]].copy()
        benunit_outputs = sim.output_dataset.data.benunit[
            ["benunit_id", *benunit_variables]
        ]
        merged_benunits = raw_benunits.merge(
            benunit_outputs,
            on="benunit_id",
            how="left",
            validate="one_to_one",
        )
        benunit_records = table_records(merged_benunits)
        selected_benunit_ids = ()
        if person_ids:
            selected_benunit_ids = tuple(
                dict.fromkeys(
                    int(row_value(row, "person_benunit_id"))
                    for row in selected
                    if row_value(row, "person_benunit_id") is not None
                )
            )
        selected_benunit_indices = select_benunit_indices(
            benunit_records,
            sample_size=sample_size,
            benunit_ids=selected_benunit_ids,
        )
        selected_benunits = [
            benunit_records[index] for index in selected_benunit_indices
        ]
    return {
        "persons": selected,
        "person_ids": [int(row_value(row, "person_id")) for row in selected],
        "benunits": selected_benunits,
        "benunit_ids": [int(row_value(row, "benunit_id")) for row in selected_benunits],
    }


def resolve_policyengine_uk_dataset_reference(dataset: str) -> str:
    if "://" in dataset:
        return dataset
    try:
        from policyengine.provenance.manifest import resolve_dataset_reference
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc
    return resolve_dataset_reference("uk", dataset)


def local_policyengine_uk_dataset(
    *,
    dataset: str,
    year: int,
) -> Any | None:
    local_path = local_policyengine_uk_dataset_path(dataset)
    if local_path is None:
        return None
    try:
        import pandas as pd
        from microdf import MicroDataFrame
        from policyengine.tax_benefit_models.uk.datasets import (
            PolicyEngineUKDataset,
            UKYearData,
        )
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc

    with pd.HDFStore(local_path, mode="r") as store:
        raw_person = store["person"].copy()
        raw_benunit = store["benunit"].copy()
        household = store["household"].copy()

    person = raw_person.merge(
        household[["household_id", "household_weight"]],
        left_on="person_household_id",
        right_on="household_id",
        how="left",
    )
    person = person.rename(columns={"household_weight": "person_weight"}).drop(
        columns=["household_id"]
    )
    benunit_household_map = person[
        ["person_benunit_id", "person_household_id"]
    ].drop_duplicates()
    benunit = raw_benunit.merge(
        benunit_household_map,
        left_on="benunit_id",
        right_on="person_benunit_id",
        how="left",
    )
    benunit = benunit.merge(
        household[["household_id", "household_weight"]],
        left_on="person_household_id",
        right_on="household_id",
        how="left",
    )
    benunit = benunit.rename(columns={"household_weight": "benunit_weight"}).drop(
        columns=[
            "person_benunit_id",
            "person_household_id",
            "household_id",
        ],
        errors="ignore",
    )

    dataset_id = Path(local_path).stem
    return PolicyEngineUKDataset(
        id=f"{dataset_id}_local_{year}",
        name=f"{dataset_id}-local-{year}",
        description=f"Local UK Dataset for year {year} based on {dataset_id}",
        filepath=str(Path(".axiom") / "policyengine-data" / f"{dataset_id}_{year}.h5"),
        year=int(year),
        data=UKYearData(
            person=MicroDataFrame(person, weights="person_weight"),
            benunit=MicroDataFrame(benunit, weights="benunit_weight"),
            household=MicroDataFrame(household, weights="household_weight"),
        ),
    )


def local_policyengine_uk_dataset_path(dataset: str) -> Path | None:
    direct_path = Path(dataset).expanduser()
    if direct_path.exists():
        return direct_path.resolve()
    try:
        from policyengine.provenance.manifest import (
            resolve_dataset_reference,
            resolve_local_managed_dataset_source,
        )
    except ImportError:
        return None

    resolved_dataset = dataset
    if "://" not in dataset:
        try:
            resolved_dataset = resolve_dataset_reference("uk", dataset)
        except ValueError:
            return None
    local_source = resolve_local_managed_dataset_source("uk", resolved_dataset)
    local_path = Path(local_source).expanduser()
    if local_path.exists():
        return local_path.resolve()
    return None


def policyengine_person_variables_for_surfaces(
    surfaces: list[str],
) -> tuple[str, ...]:
    return policyengine_variables_for_surfaces(surfaces, entity="person")


def policyengine_benunit_variables_for_surfaces(
    surfaces: list[str],
) -> tuple[str, ...]:
    return policyengine_variables_for_surfaces(surfaces, entity="benunit")


def policyengine_variables_for_surfaces(
    surfaces: list[str],
    *,
    entity: str,
) -> tuple[str, ...]:
    variables: set[str] = set()
    for surface in surfaces:
        spec = SURFACE_SPECS[surface]
        if spec.entity == entity:
            variables.update(spec.pe_variables)
    return tuple(sorted(variables))


def select_person_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    person_ids: tuple[int, ...] = (),
) -> list[int]:
    return select_entity_indices(
        rows,
        sample_size=sample_size,
        requested_ids=person_ids,
        id_column="person_id",
        weight_column="person_weight",
        entity_label="EFRS person_id",
    )


def select_benunit_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    benunit_ids: tuple[int, ...] = (),
) -> list[int]:
    return select_entity_indices(
        rows,
        sample_size=sample_size,
        requested_ids=benunit_ids,
        id_column="benunit_id",
        weight_column="benunit_weight",
        entity_label="EFRS benunit_id",
    )


def select_entity_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    requested_ids: tuple[int, ...] = (),
    id_column: str,
    weight_column: str,
    entity_label: str,
) -> list[int]:
    eligible = [
        index
        for index, row in enumerate(rows)
        if money(row_value(row, weight_column, 0)) > 0
    ]
    if not requested_ids:
        return eligible if sample_size <= 0 else eligible[:sample_size]

    requested_ids = tuple(dict.fromkeys(int(value) for value in requested_ids))
    index_by_id = {
        int(row_value(row, id_column)): index for index, row in enumerate(rows)
    }
    eligible_set = set(eligible)
    selected: list[int] = []
    missing: list[int] = []
    filtered: list[int] = []
    for person_id in requested_ids:
        index = index_by_id.get(person_id)
        if index is None:
            missing.append(person_id)
        elif index not in eligible_set:
            filtered.append(person_id)
        else:
            selected.append(index)
    if missing:
        raise SystemExit(
            f"Requested {entity_label} not found: "
            + ", ".join(str(value) for value in missing)
        )
    if filtered:
        raise SystemExit(
            f"Requested {entity_label} is not eligible for this comparison: "
            + ", ".join(str(value) for value in filtered)
        )
    return selected


def run_axiom_surface(
    *,
    program: Path,
    request: dict[str, Any],
    rulespec_root: Path,
    axiom_rules_path: Path,
    surface: str,
) -> list[dict[str, Any]]:
    if surface.startswith("universal-credit-"):
        return run_axiom_parameter_outputs(program=program, request=request)
    return run_axiom_program(
        program=program,
        request=request,
        rulespec_root=rulespec_root,
        axiom_rules_path=axiom_rules_path,
    )


def run_axiom_parameter_outputs(
    *,
    program: Path,
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    parameter_cache: dict[str, dict[str, float]] = {}
    for query in request.get("queries", []):
        period_start = str((query.get("period") or {}).get("start") or "")
        if period_start not in parameter_cache:
            parameter_cache[period_start] = rulespec_scalar_parameter_values(
                program,
                period_start=period_start,
            )
        parameter_values = parameter_cache[period_start]
        outputs: dict[str, dict[str, Any]] = {}
        for output in query.get("outputs", []):
            if output not in parameter_values:
                raise SystemExit(f"unknown RuleSpec scalar parameter: {output}")
            outputs[output] = {"value": {"value": str(parameter_values[output])}}
        results.append({"outputs": outputs})
    return results


def rulespec_scalar_parameter_values(
    program: Path, *, period_start: str
) -> dict[str, float]:
    payload = yaml.safe_load(program.read_text()) or {}
    base = rule_base_from_program(program)
    values: dict[str, float] = {}
    for rule in payload.get("rules") or []:
        if str(rule.get("kind") or "").strip() != "parameter":
            continue
        version = effective_parameter_version(rule.get("versions") or [], period_start)
        if version is None:
            continue
        formula = str(version.get("formula") or "").strip().replace("_", "")
        try:
            value = float(formula)
        except ValueError:
            continue
        values[f"{base}#{rule['name']}"] = value
    return values


def effective_parameter_version(
    versions: list[dict[str, Any]],
    period_start: str,
) -> dict[str, Any] | None:
    eligible = [
        version
        for version in versions
        if str(version.get("effective_from") or "") <= period_start
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda version: str(version.get("effective_from") or ""))


def rule_base_from_program(program: Path) -> str:
    parts = program.with_suffix("").parts
    if "regulations" in parts:
        index = parts.index("regulations")
        return "uk:" + "/".join(parts[index:])
    if "statutes" in parts:
        index = parts.index("statutes")
        return "uk:" + "/".join(parts[index:])
    raise ValueError(f"cannot infer UK RuleSpec base from {program}")


def build_axiom_request(
    *,
    pe_data: dict[str, Any],
    year: int,
    surface: str = "personal-allowance",
) -> dict[str, Any]:
    if surface == "personal-allowance":
        return build_personal_allowance_request(pe_data=pe_data, year=year)
    if surface == "child-benefit":
        return build_child_benefit_request(pe_data=pe_data, year=year)
    if surface == "pension-credit":
        return build_pension_credit_request(pe_data=pe_data, year=year)
    if surface.startswith("universal-credit-"):
        return build_universal_credit_request(
            pe_data=pe_data,
            year=year,
            surface=surface,
        )
    raise ValueError(f"unsupported UK EFRS surface: {surface}")


def build_personal_allowance_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = tax_year_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "personal-allowance"):
        entity_id = person_entity_id(int(row_value(row, "person_id")))
        for name, value in project_personal_allowance_inputs(row).items():
            inputs.append(
                input_record(
                    f"{PERSONAL_ALLOWANCE_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"] for spec in PERSONAL_ALLOWANCE_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_child_benefit_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_week_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "child-benefit"):
        entity_id = person_entity_id(int(row_value(row, "person_id")))
        for name, value in project_child_benefit_inputs(row).items():
            inputs.append(
                input_record(
                    f"{CHILD_BENEFIT_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [spec["axiom"] for spec in CHILD_BENEFIT_OUTPUTS.values()],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_pension_credit_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_week_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "pension-credit"):
        entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        for name, value in project_pension_credit_inputs(row).items():
            inputs.append(
                input_record(
                    f"{PENSION_CREDIT_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [spec["axiom"] for spec in PENSION_CREDIT_OUTPUTS.values()],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_universal_credit_request(
    *, pe_data: dict[str, Any], year: int, surface: str
) -> dict[str, Any]:
    interval = benefit_month_interval(year)
    spec = SURFACE_SPECS[surface]
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, surface):
        if spec.entity == "benunit":
            entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        else:
            entity_id = person_entity_id(int(row_value(row, "person_id")))
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [output["axiom"] for output in spec.outputs.values()],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": [], "relations": []},
        "queries": queries,
    }


def project_personal_allowance_inputs(row: Any) -> dict[str, Any]:
    adjusted_net_income = money(row_value(row, "adjusted_net_income"))
    gift_aid_grossed_up = money(row_value(row, "gift_aid_grossed_up", 0))
    return {
        "individual_makes_claim": True,
        "individual_meets_requirements_under_section_56": True,
        "adjusted_net_income": max(0.0, adjusted_net_income - gift_aid_grossed_up),
    }


def project_child_benefit_inputs(row: Any) -> dict[str, Any]:
    child_index = int(row_value(row, "child_benefit_child_index", -1))
    is_eldest = child_index == 1
    return {
        "during_subsistence_of_marriage_any_party_married_to_more_than_one_person": False,
        "marriage_ceremony_took_place_under_law_permitting_polygamy": False,
        "specified_benefit_allowance_or_increase_paid_for_week_to_person": False,
        "specified_benefit_is_in_respect_of_only_elder_or_eldest_child_for_child_benefit_entitlement": False,
        "child_or_qualifying_young_person_is_only_elder_or_eldest_for_payee": is_eldest,
        "paragraph_2_relationship_coordination_applies": False,
        "child_or_qualifying_young_person_is_elder_or_eldest_among_paragraph_2_children": is_eldest,
        "payee_is_voluntary_organisation": False,
        "payee_resides_with_parent_otherwise_than_paragraph_2_a": False,
    }


def project_pension_credit_inputs(row: Any) -> dict[str, Any]:
    relation_type = str(row_value(row, "relation_type", "")).upper()
    is_couple = bool(row_value(row, "is_couple", False)) or relation_type == "COUPLE"
    severe_disability_addition = (
        money(row_value(row, "severe_disability_minimum_guarantee_addition", 0))
        / WEEKS_IN_YEAR
    )
    num_carers = int(money(row_value(row, "num_carers", 0)))
    return {
        "claimant_is_prisoner": False,
        "member_of_religious_order_fully_maintained_by_order": False,
        "claimant_has_partner": is_couple,
        "treated_as_severely_disabled_person_under_schedule_i_part_i_paragraph_1": severe_disability_addition
        > 0,
        "severe_disability_couple_rate_conditions_satisfied": severe_disability_addition
        > 100,
        "paragraph_4_of_part_ii_of_schedule_i_satisfied_for_this_partner": num_carers
        > 0,
    }


def rows_for_surface(pe_data: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    persons = pe_data["persons"]
    if surface == "child-benefit":
        return [
            row
            for row in persons
            if money(row_value(row, "child_benefit_respective_amount", 0)) > 0
        ]
    if surface == "universal-credit-child-element":
        return [
            row
            for row in persons
            if money(row_value(row, "uc_individual_child_element", 0)) > 0
            or money(row_value(row, "uc_individual_disabled_child_element", 0)) > 0
            or money(row_value(row, "uc_individual_severely_disabled_child_element", 0))
            > 0
        ]
    benunits = pe_data.get("benunits", [])
    if surface == "universal-credit-standard-allowance":
        return [
            row
            for row in benunits
            if money(row_value(row, "uc_standard_allowance", 0)) > 0
        ]
    if surface == "universal-credit-lcwra-element":
        return [
            row for row in benunits if money(row_value(row, "uc_LCWRA_element", 0)) > 0
        ]
    if surface == "universal-credit-carer-element":
        return [
            row for row in benunits if money(row_value(row, "uc_carer_element", 0)) > 0
        ]
    if surface == "universal-credit-childcare-cap":
        return [
            row
            for row in benunits
            if money(row_value(row, "uc_maximum_childcare_element_amount", 0)) > 0
        ]
    if SURFACE_SPECS[surface].entity == "benunit":
        return benunits
    return persons


def compare_outputs(
    *,
    pe_data: dict[str, Any],
    axiom_outputs_by_surface: dict[str, list[dict[str, Any]]],
    tolerance: float,
    relative_tolerance: float,
) -> UKEFRSComparisonReport:
    mismatches: list[UKEFRSComparisonRow] = []
    oracle_divergences: list[UKEFRSOracleDivergence] = []
    summary: dict[str, dict[str, Any]] = {
        f"{surface}:{name}": {
            "surface": surface,
            "output": name,
            "compared": 0,
            "mismatches": 0,
            "oracle_divergences": 0,
            "max_abs_diff": 0.0,
            "max_relative_diff": 0.0,
        }
        for surface, spec in SURFACE_SPECS.items()
        if surface in axiom_outputs_by_surface
        for name in spec.outputs
    }
    compared_values = 0
    for surface, axiom_outputs in axiom_outputs_by_surface.items():
        output_specs = SURFACE_SPECS[surface].outputs
        persons = rows_for_surface(pe_data, surface)
        for index, result in enumerate(axiom_outputs):
            pe_row = persons[index]
            entity_id = entity_id_for_surface(surface, pe_row)
            outputs = result.get("outputs") or {}
            for name, spec in output_specs.items():
                if not output_applies(spec, pe_row):
                    continue
                axiom_value = output_number(outputs.get(spec["axiom"]))
                pe_value = policyengine_output_value(spec, pe_row)
                diff = axiom_value - pe_value
                abs_diff = abs(diff)
                compared_values += 1
                summary_key = f"{surface}:{name}"
                summary[summary_key]["compared"] += 1
                summary[summary_key]["max_abs_diff"] = max(
                    summary[summary_key]["max_abs_diff"], abs_diff
                )
                summary[summary_key]["max_relative_diff"] = max(
                    summary[summary_key]["max_relative_diff"],
                    relative_diff(axiom_value, pe_value),
                )
                if not within_tolerance(
                    axiom_value,
                    pe_value,
                    absolute_tolerance=tolerance,
                    relative_tolerance=relative_tolerance,
                ):
                    divergence = known_policyengine_divergence(
                        surface=surface,
                        output=name,
                        entity_id=entity_id,
                        axiom_value=axiom_value,
                        policyengine_value=pe_value,
                        diff=diff,
                    )
                    if divergence is not None:
                        summary[summary_key]["oracle_divergences"] += 1
                        oracle_divergences.append(divergence)
                    else:
                        summary[summary_key]["mismatches"] += 1
                        mismatches.append(
                            UKEFRSComparisonRow(
                                surface=surface,
                                entity_id=entity_id,
                                output=name,
                                axiom=axiom_value,
                                policyengine=pe_value,
                                diff=diff,
                            )
                        )
    return UKEFRSComparisonReport(
        compared_persons=len(pe_data["person_ids"]),
        compared_benunits=len(pe_data.get("benunit_ids", [])),
        compared_values=compared_values,
        mismatches=mismatches,
        oracle_divergences=oracle_divergences,
        output_summary=list(summary.values()),
        skipped_surfaces=SKIPPED_SURFACES,
        projection_notes=[
            "Personal allowance projection supplies EFRS adjusted net income "
            "net of PolicyEngine's gift_aid_grossed_up taper adjustment, because "
            "PolicyEngine UK applies that subtraction inside its personal "
            "allowance formula.",
            "The current projection treats EFRS people as making a claim and "
            "meeting the Section 56 residence/citizenship-condition boundary "
            "facts, matching the usual PolicyEngine UK EFRS personal allowance "
            "surface until those upstream legal predicates are encoded.",
            "Child Benefit comparison filters to positive PolicyEngine "
            "child_benefit_respective_amount rows, divides that annualized "
            "PolicyEngine output by 52 to compare against the RuleSpec weekly "
            "rate, and projects the eldest-child branch from "
            "child_benefit_child_index.",
            "Child Benefit relationship-coordination, voluntary-organisation, "
            "specified-benefit, and polygamous-marriage branches are projected "
            "false because PolicyEngine UK's child_benefit_respective_amount "
            "does not expose those legal predicates separately.",
            "Pension Credit standard minimum guarantee comparison runs at "
            "benefit-unit level, divides PolicyEngine's annual output by 52, "
            "and projects claimant_has_partner from PolicyEngine's relation_type "
            "or is_couple. Prisoner and fully-maintained religious-order branches "
            "are projected false because those legal predicates are not exposed "
            "in the EFRS oracle data.",
            "Pension Credit carer additions compare RuleSpec's per-partner "
            "amount against PolicyEngine's annual aggregate carer addition "
            "divided by num_carers and 52. The EFRS oracle has no positive "
            "severe-disability addition rows, so that branch is currently a "
            "zero-row guard rather than a positive-eligibility validation.",
            "Universal Credit Regulation 36 comparisons treat the generated "
            "RuleSpec outputs as component table amounts. PolicyEngine annual "
            "EFRS component outputs are divided by 12, and EFRS category "
            "variables select the matching standard-allowance, child-element, "
            "carer, LCWRA, and childcare-cap rows.",
        ],
    )


def entity_id_for_surface(surface: str, row: Any) -> str:
    if SURFACE_SPECS[surface].entity == "benunit":
        return benunit_entity_id(int(row_value(row, "benunit_id")))
    return person_entity_id(int(row_value(row, "person_id")))


def policyengine_output_value(spec: dict[str, Any], row: Any) -> float:
    raw_value = money(row_value(row, spec["pe"]))
    if spec.get("pe_transform") == "annual_to_weekly":
        return raw_value / WEEKS_IN_YEAR
    if spec.get("pe_transform") == "annual_to_weekly_per_carer":
        return (
            raw_value
            / WEEKS_IN_YEAR
            / max(1, int(money(row_value(row, "num_carers", 0))))
        )
    if spec.get("pe_transform") == "annual_to_monthly":
        return raw_value / MONTHS_IN_YEAR
    return raw_value


def output_applies(spec: dict[str, Any], row: Any) -> bool:
    applies = spec.get("applies")
    if applies is None:
        return True
    if applies == "positive_pe_output":
        return policyengine_output_value(spec, row) > 0
    if applies == "uc_first_child_element":
        return (
            policyengine_output_value(spec, row) > 0
            and int(row_value(row, "uc_child_index", -1)) == 1
            and bool(row_value(row, "uc_is_child_born_before_child_limit", False))
        )
    if applies == "uc_subsequent_child_element":
        return policyengine_output_value(spec, row) > 0 and not output_applies(
            {**spec, "applies": "uc_first_child_element"},
            row,
        )
    if applies == "uc_lcwra_standard_amount":
        monthly_value = policyengine_output_value(spec, row)
        return 0 < monthly_value < 300
    if applies == "uc_lcwra_higher_amount":
        monthly_value = policyengine_output_value(spec, row)
        return 300 <= monthly_value < 600
    if applies == "uc_childcare_two_or_more_children":
        return int(row_value(row, "uc_childcare_element_eligible_children", 0)) >= 2
    if isinstance(applies, tuple) and len(applies) == 2:
        name, expected = applies
        value = row_value(row, name)
        if isinstance(expected, str):
            return enum_name(value) == expected
        return value == expected
    raise ValueError(f"unsupported output applicability rule: {applies!r}")


def enum_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def known_policyengine_divergence(
    *,
    surface: str,
    output: str,
    entity_id: str,
    axiom_value: float,
    policyengine_value: float,
    diff: float,
) -> UKEFRSOracleDivergence | None:
    if (
        surface == "personal-allowance"
        and output == "personal_allowance"
        and 0 < diff < 1
        and math.isclose(axiom_value, math.ceil(policyengine_value), abs_tol=1e-9)
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently returns fractional tapered personal "
                "allowances instead of rounding up to the nearest pound under "
                "ITA 2007 s.35(3)."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1738",
        )
    if (
        surface == "child-benefit"
        and output == "child_benefit_weekly_rate"
        and 0 < diff < 0.2
        and (
            math.isclose(axiom_value, 27.05, abs_tol=1e-9)
            or math.isclose(axiom_value, 17.90, abs_tol=1e-9)
        )
        and (
            math.isclose(policyengine_value, 26.935709699992934, abs_tol=0.005)
            or math.isclose(policyengine_value, 17.836506423219888, abs_tol=0.005)
        )
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 Child "
                "Benefit amounts instead of the published 2026-27 weekly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1739",
        )
    if (
        surface == "pension-credit"
        and output == "standard_minimum_guarantee"
        and 0 < diff < 15
        and (
            math.isclose(axiom_value, 238.00, abs_tol=1e-9)
            or math.isclose(axiom_value, 363.25, abs_tol=1e-9)
        )
        and (
            math.isclose(policyengine_value, 229.3929826081932, abs_tol=0.01)
            or math.isclose(policyengine_value, 350.12286608501375, abs_tol=0.01)
        )
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 Pension "
                "Credit guarantee amounts instead of the published 2026-27 "
                "weekly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1740",
        )
    if (
        surface == "pension-credit"
        and output in {"severe_disability_additional_amount", "carer_additional_amount"}
        and policyengine_value > 0
        and 0 < abs(diff) < 5
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 "
                "Pension Credit additional amounts instead of the published "
                "2026-27 weekly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1742",
        )
    expected_uc_rate = UNIVERSAL_CREDIT_2026_RULESPEC_RATES.get(output)
    if (
        surface.startswith("universal-credit-")
        and expected_uc_rate is not None
        and math.isclose(axiom_value, expected_uc_rate, abs_tol=1e-9)
        and policyengine_value > 0
        and 0 < abs(diff) < 75
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 "
                "Universal Credit Regulation 36 amounts instead of the "
                "published 2026-27 monthly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1741",
        )
    return None


def print_report(
    report: UKEFRSComparisonReport,
    *,
    tolerance: float,
    relative_tolerance: float,
) -> None:
    print("PolicyEngine UK EFRS comparison")
    print(f"Compared persons: {report.compared_persons:,}")
    print(f"Compared benefit units: {report.compared_benunits:,}")
    print(f"Compared values: {report.compared_values:,}")
    print(f"Tolerance: {tolerance:g}")
    print(f"Relative tolerance: {relative_tolerance:g}")
    print(f"Mismatches: {len(report.mismatches):,}")
    print(f"Known PolicyEngine oracle divergences: {len(report.oracle_divergences):,}")
    print()
    print("By output:")
    for item in report.output_summary:
        print(
            f"  - {item['surface']}:{item['output']}: "
            f"{item['mismatches']:,}/{item['compared']:,} mismatch, "
            f"{item['oracle_divergences']:,} known PE divergence, "
            f"max_abs_diff={item['max_abs_diff']:.2f}, "
            f"max_rel_diff={item['max_relative_diff']:.2g}"
        )
    if report.mismatches:
        print()
        print("Top mismatches:")
        for row in sorted(
            report.mismatches, key=lambda item: abs(item.diff), reverse=True
        )[:20]:
            print(
                f"  - entity={row.entity_id} {row.surface}:{row.output}: "
                f"axiom={row.axiom:.2f} pe={row.policyengine:.2f} "
                f"diff={row.diff:.2f}"
            )
    if report.oracle_divergences:
        print()
        print("Known PolicyEngine oracle divergences:")
        for row in sorted(
            report.oracle_divergences, key=lambda item: abs(item.diff), reverse=True
        )[:20]:
            print(
                f"  - entity={row.entity_id} {row.surface}:{row.output}: "
                f"axiom={row.axiom:.2f} pe={row.policyengine:.2f} "
                f"diff={row.diff:.2f}; {row.issue_url}"
            )
    if report.skipped_surfaces:
        print()
        print("Skipped mapped UK surfaces:")
        for item in report.skipped_surfaces:
            print(f"  - {item['surface']}: {item['reason']}")
    print()
    print("Projection notes:")
    for note in report.projection_notes:
        print(f"  - {note}")


def table_records(table: Any) -> list[dict[str, Any]]:
    if isinstance(table, list):
        return [dict(row) for row in table]
    if isinstance(table, dict):
        keys = list(table)
        length = len(table[keys[0]]) if keys else 0
        return [{key: table[key][index] for key in keys} for index in range(length)]
    if hasattr(table, "to_dict"):
        return table.to_dict("records")
    raise TypeError(f"unsupported table type: {type(table).__name__}")


def row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    if hasattr(row, "get"):
        return row.get(name, default)
    return getattr(row, name, default)


def tax_year_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }


def benefit_week_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "custom",
        "name": "benefit_week",
        "start": f"{year:04d}-04-06",
        "end": f"{year:04d}-04-12",
    }


def benefit_month_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "month",
        "name": "benefit_month",
        "start": f"{year:04d}-04-01",
        "end": f"{year:04d}-04-30",
    }


def person_entity_id(person_id: int) -> str:
    return f"person_{person_id}"


def benunit_entity_id(benunit_id: int) -> str:
    return f"benunit_{benunit_id}"


def resolve_workspace_root(root: Path | None) -> Path:
    if root is not None:
        return root.resolve()
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents, Path.home() / "TheAxiomFoundation"]:
        if (candidate / "rulespec-uk").exists() and (
            candidate / "axiom-rules-engine"
        ).exists():
            return candidate
    return cwd


def policyengine_uk_install_message() -> str:
    return (
        "Run with: uv run "
        f"--with 'policyengine[uk]=={POLICYENGINE_VERSION}' "
        "axiom-encode uk-efrs-compare"
    )


def log(message: str) -> None:
    print(message, file=sys.stderr)

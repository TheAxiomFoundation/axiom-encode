"""Compare federal tax RuleSpec output against PolicyEngine ECPS.

This is the federal-tax counterpart to ``snap-ecps-compare``. Each comparison
surface has an explicit projection from ECPS fields into Axiom inputs. The
projection is intentionally conservative: it does not use PE outputs for the
surface under test as Axiom inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without optional oracle deps
    np = None


POLICYENGINE_VERSION = "4.4.4"
POLICYENGINE_US_VERSION = "1.691.3"
DATASET = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"

CTC_PROGRAM_PATH = Path("statutes/26/24.yaml")
CTC_TEST_PATH = Path("statutes/26/24.test.yaml")
CTC_BASE = "us:statutes/26/24"
CTC_H_BASE = "us:statutes/26/24/h"
STANDARD_DEDUCTION_PROGRAM_PATH = Path(
    "policies/irs/rev-proc-2025-32/standard-deduction.yaml"
)
STANDARD_DEDUCTION_BASE = "us:policies/irs/rev-proc-2025-32/standard-deduction"
EMPLOYEE_OASDI_PROGRAM_PATH = Path("statutes/26/3101/a.yaml")
EMPLOYEE_OASDI_BASE = "us:statutes/26/3101/a"
EMPLOYEE_MEDICARE_PROGRAM_PATH = Path("statutes/26/3101/b/1.yaml")
EMPLOYEE_MEDICARE_BASE = "us:statutes/26/3101/b/1"
EMPLOYER_OASDI_PROGRAM_PATH = Path("statutes/26/3111/a.yaml")
EMPLOYER_OASDI_BASE = "us:statutes/26/3111/a"
EMPLOYER_MEDICARE_PROGRAM_PATH = Path("statutes/26/3111/b.yaml")
EMPLOYER_MEDICARE_BASE = "us:statutes/26/3111/b"
OASDI_WAGE_BASE_PROGRAM_PATH = Path("statutes/26/3121/a/1.yaml")
OASDI_WAGE_BASE_BASE = "us:statutes/26/3121/a/1"
OASDI_WAGE_BASE_EXCLUSION_OUTPUT = (
    f"{OASDI_WAGE_BASE_BASE}#oasdi_wage_base_excess_excluded_remuneration"
)
AXIOM_3121_TAXABLE_OASDI_WAGES = "axiom_3121_taxable_oasdi_wages"
CAPITAL_GAINS_PROGRAM_PATH = Path("statutes/26/1/h.yaml")
CAPITAL_GAINS_BASE = "us:statutes/26/1/h"
EITC_PROGRAM_PATH = Path("statutes/26/32.yaml")
EITC_BASE = "us:statutes/26/32"
SECTION_112_BASE = "us:statutes/26/112"
SECTION_32_C_2_BASE = "us:statutes/26/32/c/2"
SECTION_152_C_BASE = "us:statutes/26/152/c"
SECTION_164_F_BASE = "us:statutes/26/164/f"
SECTION_1401_BASE = "us:statutes/26/1401"
SECTION_1402_A_BASE = "us:statutes/26/1402/a"


def contribution_and_benefit_base_program_path(year: int) -> Path:
    return Path(f"policies/ssa/contribution-and-benefit-base/{year}.yaml")


def contribution_and_benefit_base_base(year: int) -> str:
    return f"us:policies/ssa/contribution-and-benefit-base/{year}"


def contribution_and_benefit_base_output(year: int) -> str:
    return (
        f"{contribution_and_benefit_base_base(year)}"
        "#contribution_and_benefit_base_under_section_230_of_social_security_act"
    )


CTC_OUTPUTS = {
    "ctc_before_advance_payments": {
        "axiom": f"{CTC_BASE}#ctc_before_advance_payments",
        "pe": "ctc",
    },
    "ctc_maximum_before_phaseout": {
        "axiom": f"{CTC_BASE}#ctc_maximum_before_phaseout",
        "pe": "ctc_maximum",
    },
    "ctc_phaseout_threshold": {
        "axiom": f"{CTC_BASE}#ctc_phaseout_threshold",
        "pe": "ctc_phase_out_threshold",
    },
    "ctc_phaseout_amount": {
        "axiom": f"{CTC_BASE}#ctc_phaseout_amount",
        "pe": "ctc_phase_out",
    },
    "ctc_qualifying_children_count": {
        "axiom": f"{CTC_BASE}#ctc_qualifying_children_count",
        "pe": "ctc_qualifying_children",
    },
}
STANDARD_DEDUCTION_OUTPUTS = {
    "basic_standard_deduction": {
        "axiom": f"{STANDARD_DEDUCTION_BASE}#basic_standard_deduction_after_dependent_limit",
        "pe": "basic_standard_deduction",
    },
    "additional_standard_deduction": {
        "axiom": f"{STANDARD_DEDUCTION_BASE}#additional_standard_deduction_for_aged_or_blind",
        "pe": "additional_standard_deduction",
    },
    "standard_deduction": {
        "axiom": f"{STANDARD_DEDUCTION_BASE}#standard_deduction",
        "pe": "standard_deduction",
    },
}
PAYROLL_SURFACES = {
    "employee-oasdi": {
        "program": EMPLOYEE_OASDI_PROGRAM_PATH,
        "base": EMPLOYEE_OASDI_BASE,
        "wage_input": AXIOM_3121_TAXABLE_OASDI_WAGES,
        "outputs": {
            "employee_social_security_tax": {
                "axiom": f"{EMPLOYEE_OASDI_BASE}#oasdi_wage_tax",
                "pe": "employee_social_security_tax",
            },
        },
    },
    "employee-medicare": {
        "program": EMPLOYEE_MEDICARE_PROGRAM_PATH,
        "base": EMPLOYEE_MEDICARE_BASE,
        "wage_input": "payroll_tax_gross_wages",
        "outputs": {
            "employee_medicare_tax": {
                "axiom": f"{EMPLOYEE_MEDICARE_BASE}#hospital_insurance_wage_tax",
                "pe": "employee_medicare_tax",
            },
        },
    },
    "employer-oasdi": {
        "program": EMPLOYER_OASDI_PROGRAM_PATH,
        "base": EMPLOYER_OASDI_BASE,
        "wage_input": AXIOM_3121_TAXABLE_OASDI_WAGES,
        "outputs": {
            "employer_social_security_tax": {
                "axiom": f"{EMPLOYER_OASDI_BASE}#employer_oasdi_excise_tax",
                "pe": "employer_social_security_tax",
            },
        },
    },
    "employer-medicare": {
        "program": EMPLOYER_MEDICARE_PROGRAM_PATH,
        "base": EMPLOYER_MEDICARE_BASE,
        "wage_input": "payroll_tax_gross_wages",
        "outputs": {
            "employer_medicare_tax": {
                "axiom": f"{EMPLOYER_MEDICARE_BASE}#hospital_insurance_employer_tax",
                "pe": "employer_medicare_tax",
            },
        },
    },
}
CAPITAL_GAINS_DEFINITION_OUTPUTS = {
    "net_capital_gain": {
        "axiom": f"{CAPITAL_GAINS_BASE}#net_capital_gain",
        "pe": "net_capital_gain",
    },
    "adjusted_net_capital_gain": {
        "axiom": f"{CAPITAL_GAINS_BASE}#adjusted_net_capital_gain",
        "pe": "adjusted_net_capital_gain",
    },
}
EITC_OUTPUTS = {
    "eitc_child_count": {
        "axiom": f"{EITC_BASE}#eitc_child_count",
        "pe": "eitc_child_count",
    },
    "eitc_phase_in_rate": {
        "axiom": f"{EITC_BASE}#eitc_phase_in_rate",
        "pe": "eitc_phase_in_rate",
    },
    "eitc_phase_out_rate": {
        "axiom": f"{EITC_BASE}#eitc_phase_out_rate",
        "pe": "eitc_phase_out_rate",
    },
    "eitc_maximum": {
        "axiom": f"{EITC_BASE}#eitc_maximum",
        "pe": "eitc_maximum",
    },
    "eitc_phase_out_start": {
        "axiom": f"{EITC_BASE}#eitc_phase_out_start",
        "pe": "eitc_phase_out_start",
    },
    "eitc_phased_in": {
        "axiom": f"{EITC_BASE}#eitc_phased_in",
        "pe": "eitc_phased_in",
    },
    "eitc_reduction": {
        "axiom": f"{EITC_BASE}#eitc_reduction",
        "pe": "eitc_reduction",
    },
    "eitc_investment_income_eligible": {
        "axiom": f"{EITC_BASE}#eitc_investment_income_eligible",
        "pe": "eitc_investment_income_eligible",
    },
    "eitc_demographic_eligible": {
        "axiom": f"{EITC_BASE}#eitc_demographic_eligible",
        "pe": "eitc_demographic_eligible",
    },
    "eitc_eligible": {
        "axiom": f"{EITC_BASE}#eitc_allowed",
        "pe": "eitc_eligible",
    },
}
SURFACE_OUTPUTS = {
    "ctc": CTC_OUTPUTS,
    "standard-deduction": STANDARD_DEDUCTION_OUTPUTS,
    "capital-gain-definitions": CAPITAL_GAINS_DEFINITION_OUTPUTS,
    "eitc": EITC_OUTPUTS,
    **{surface: config["outputs"] for surface, config in PAYROLL_SURFACES.items()},
}
SURFACE_PROGRAM_PATHS = {
    "ctc": CTC_PROGRAM_PATH,
    "standard-deduction": STANDARD_DEDUCTION_PROGRAM_PATH,
    "capital-gain-definitions": CAPITAL_GAINS_PROGRAM_PATH,
    "eitc": EITC_PROGRAM_PATH,
    **{surface: config["program"] for surface, config in PAYROLL_SURFACES.items()},
}
PE_TAX_UNIT_VARIABLES = tuple(
    sorted(
        {
            "additional_standard_deduction",
            "adjusted_gross_income",
            "adjusted_net_capital_gain",
            "basic_standard_deduction",
            "ctc",
            "ctc_maximum",
            "ctc_phase_out",
            "ctc_phase_out_threshold",
            "ctc_qualifying_children",
            "eitc_child_count",
            "eitc_demographic_eligible",
            "eitc_eligible",
            "eitc_investment_income_eligible",
            "eitc_maximum",
            "eitc_phase_in_rate",
            "eitc_phase_out_rate",
            "eitc_phase_out_start",
            "eitc_phased_in",
            "eitc_reduction",
            "filing_status",
            "net_capital_gain",
            "standard_deduction",
        }
    )
)
PE_PERSON_VARIABLES = tuple(
    sorted(
        {
            "employee_medicare_tax",
            "employee_social_security_tax",
            "employer_medicare_tax",
            "employer_social_security_tax",
            "irs_employment_income",
            "payroll_tax_gross_wages",
        }
    )
)

FILING_STATUS_CODES = {
    "SINGLE": 0,
    "JOINT": 1,
    "SEPARATE": 2,
    "HEAD_OF_HOUSEHOLD": 3,
    "SURVIVING_SPOUSE": 4,
}
VALID_CHILD_SSN_TYPES = {"CITIZEN", "NON_CITIZEN_VALID_EAD"}


@dataclass(frozen=True)
class TaxComparisonRow:
    surface: str
    entity_id: str
    output: str
    axiom: float
    policyengine: float
    diff: float


@dataclass(frozen=True)
class TaxComparisonReport:
    compared_tax_units: int
    compared_persons: int
    compared_values: int
    mismatches: list[TaxComparisonRow]
    output_summary: list[dict[str, Any]]
    projection_notes: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "compared_tax_units": self.compared_tax_units,
            "compared_persons": self.compared_persons,
            "compared_values": self.compared_values,
            "mismatch_count": len(self.mismatches),
            "mismatches": [row.__dict__ for row in self.mismatches],
            "output_summary": self.output_summary,
            "projection_notes": self.projection_notes,
        }


@dataclass(frozen=True)
class PersonProjectionContext:
    is_head: bool
    is_spouse: bool
    is_tax_unit_dependent: bool
    qualifying_child_under_section_152_c: bool
    qualifying_child_described_in_subsection_c: bool
    has_valid_child_ssn: bool
    filer_has_valid_child_ctc_ssn: bool


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root containing rulespec-us and axiom-rules-engine",
    )
    parser.add_argument(
        "--rulespec-root",
        type=Path,
        default=None,
        help="rulespec-us checkout; defaults to <root>/rulespec-us",
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
            "Number of positive-weight ECPS tax units to compare; "
            "0 compares all eligible tax units"
        ),
    )
    parser.add_argument(
        "--positive-ctc-only",
        action="store_true",
        help="Restrict to ECPS tax units with positive PolicyEngine CTC",
    )
    parser.add_argument(
        "--surface",
        choices=["all", *SURFACE_OUTPUTS],
        default="all",
        help="Federal tax surface to compare; defaults to all supported surfaces",
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
            "ordinary dollar outputs remain controlled by --tolerance"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--allow-policyengine-us-version",
        action="store_true",
        help=(
            "Allow the installed policyengine-us version to differ from the "
            "pinned oracle baseline. Intended only for validating local "
            "PolicyEngine fixes before release."
        ),
    )
    parser.add_argument(
        "--allow-uncertified-policyengine-data",
        action="store_true",
        help=(
            "Allow policyengine.py ECPS data to load against an overridden "
            "local policyengine-us version. Intended only with "
            "--allow-policyengine-us-version for validating local oracle fixes."
        ),
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero when any compared value differs beyond tolerance",
    )


def main(args: argparse.Namespace) -> int:
    report = compare_tax_ecps(
        workspace_root=resolve_workspace_root(args.root),
        rulespec_root=args.rulespec_root,
        axiom_rules_path=args.axiom_rules_engine_path,
        year=args.year,
        sample_size=args.sample_size,
        positive_ctc_only=args.positive_ctc_only,
        surface=args.surface,
        data_folder=args.data_folder,
        tolerance=args.tolerance,
        relative_tolerance=args.relative_tolerance,
        allow_policyengine_us_version=args.allow_policyengine_us_version,
        allow_uncertified_policyengine_data=args.allow_uncertified_policyengine_data,
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


def compare_tax_ecps(
    *,
    workspace_root: Path,
    rulespec_root: Path | None,
    axiom_rules_path: Path | None,
    year: int,
    sample_size: int,
    positive_ctc_only: bool,
    surface: str,
    data_folder: Path,
    tolerance: float,
    relative_tolerance: float,
    allow_policyengine_us_version: bool = False,
    allow_uncertified_policyengine_data: bool = False,
) -> TaxComparisonReport:
    if allow_uncertified_policyengine_data and not allow_policyengine_us_version:
        raise SystemExit(
            "--allow-uncertified-policyengine-data requires "
            "--allow-policyengine-us-version"
        )
    require_numpy()
    require_policyengine_versions(
        allow_policyengine_us_version=allow_policyengine_us_version,
    )
    resolved_rulespec_root = (rulespec_root or workspace_root / "rulespec-us").resolve()
    resolved_axiom_rules_path = (
        axiom_rules_path or workspace_root / "axiom-rules-engine"
    ).resolve()
    pe_data = load_policyengine_tax_data(
        year=year,
        sample_size=sample_size,
        positive_ctc_only=positive_ctc_only,
        data_folder=data_folder,
        allow_uncertified_policyengine_data=allow_uncertified_policyengine_data,
    )
    surfaces = list(SURFACE_OUTPUTS) if surface == "all" else [surface]
    surface_results: dict[str, list[dict[str, Any]]] = {}
    oasdi_wage_base_results: list[dict[str, Any]] | None = None
    contribution_base: float | None = None
    for selected_surface in surfaces:
        program = resolved_rulespec_root / SURFACE_PROGRAM_PATHS[selected_surface]
        if not program.exists():
            raise SystemExit(f"{selected_surface} RuleSpec not found: {program}")
        if payroll_surface_uses_axiom_3121(selected_surface):
            if contribution_base is None:
                contribution_base_program = (
                    resolved_rulespec_root
                    / contribution_and_benefit_base_program_path(year)
                )
                if not contribution_base_program.exists():
                    raise SystemExit(
                        "Contribution-and-benefit-base RuleSpec not found: "
                        f"{contribution_base_program}"
                    )
                contribution_base_results = run_axiom_program(
                    program=contribution_base_program,
                    request=build_contribution_and_benefit_base_request(year=year),
                    rulespec_root=resolved_rulespec_root,
                    axiom_rules_path=resolved_axiom_rules_path,
                )
                contribution_base = contribution_and_benefit_base_from_results(
                    contribution_base_results,
                    year=year,
                )
            if oasdi_wage_base_results is None:
                oasdi_program = resolved_rulespec_root / OASDI_WAGE_BASE_PROGRAM_PATH
                if not oasdi_program.exists():
                    raise SystemExit(
                        f"OASDI wage-base RuleSpec not found: {oasdi_program}"
                    )
                oasdi_request = build_oasdi_wage_base_request(
                    pe_data=pe_data,
                    year=year,
                    contribution_base=contribution_base,
                )
                oasdi_wage_base_results = run_axiom_program(
                    program=oasdi_program,
                    request=oasdi_request,
                    rulespec_root=resolved_rulespec_root,
                    axiom_rules_path=resolved_axiom_rules_path,
                )
        request = build_axiom_request(
            pe_data=pe_data,
            year=year,
            surface=selected_surface,
            oasdi_wage_base_results=oasdi_wage_base_results,
        )
        surface_results[selected_surface] = run_axiom_program(
            program=program,
            request=request,
            rulespec_root=resolved_rulespec_root,
            axiom_rules_path=resolved_axiom_rules_path,
        )
    return compare_outputs(
        pe_data=pe_data,
        axiom_outputs_by_surface=surface_results,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )


def load_policyengine_tax_data(
    *,
    year: int,
    sample_size: int,
    positive_ctc_only: bool,
    data_folder: Path,
    allow_uncertified_policyengine_data: bool = False,
) -> dict[str, Any]:
    if allow_uncertified_policyengine_data:
        _install_policyengine_data_certification_override()
    try:
        from policyengine.core import Simulation
        from policyengine.tax_benefit_models.us import ensure_datasets, us_latest
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_install_message()) from exc

    log("Loading PolicyEngine ECPS...")
    datasets = ensure_datasets(
        datasets=[DATASET],
        years=[year],
        data_folder=str(data_folder),
    )
    dataset = datasets[f"enhanced_cps_2024_{year}"]
    sim = Simulation(
        dataset=dataset,
        tax_benefit_model_version=us_latest,
        extra_variables={
            "person": list(PE_PERSON_VARIABLES),
            "tax_unit": list(PE_TAX_UNIT_VARIABLES),
        },
    )
    log("Running PolicyEngine tax outputs...")
    sim.run()

    tax_units = sim.output_dataset.data.tax_unit
    raw_tax_units = dataset.data.tax_unit
    tax_unit_outputs = tax_units[["tax_unit_id", *PE_TAX_UNIT_VARIABLES]].copy()
    raw_persons = dataset.data.person
    person_outputs = sim.output_dataset.data.person[
        ["person_id", *PE_PERSON_VARIABLES]
    ].copy()
    mask = np.asarray(raw_tax_units["tax_unit_weight"]) > 0
    if positive_ctc_only:
        mask &= np.asarray(tax_units["ctc"]) > 0
    indices = np.flatnonzero(mask)
    if sample_size > 0:
        indices = indices[:sample_size]
    selected = raw_tax_units.iloc[indices].copy()
    selected = selected.merge(
        tax_unit_outputs,
        on="tax_unit_id",
        how="left",
        validate="one_to_one",
    )
    selected_ids = set(int(value) for value in selected["tax_unit_id"])
    selected_persons = raw_persons[
        raw_persons["person_tax_unit_id"].astype(int).isin(selected_ids)
    ].copy()
    selected_persons = selected_persons.merge(
        person_outputs,
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    return {
        "tax_units": selected,
        "persons": selected_persons,
        "tax_unit_ids": [int(value) for value in selected["tax_unit_id"]],
        "person_ids": [int(value) for value in selected_persons["person_id"]],
    }


def _install_policyengine_data_certification_override() -> None:
    """Allow ECPS oracle runs against a local policyengine-us fix branch.

    policyengine.py 4.4.4 certifies the bundled ECPS data against
    policyengine-us 1.691.3. When validating a local policyengine-us PR, the
    installed model version can intentionally differ while the ECPS data is
    still the desired oracle input. This override is intentionally reachable
    only through an explicit CLI flag paired with --allow-policyengine-us-version.
    """
    os.environ.setdefault("POLICYENGINE_SKIP_COUNTRY_IMPORTS", "1")
    try:
        import policyengine.provenance.manifest as manifest
    except ImportError:  # pragma: no cover - optional runtime dependency
        return

    def _allow_local_oracle_data(
        country_id: str,
        runtime_model_version: str,
        runtime_data_build_fingerprint: str | None = None,
    ):
        return manifest.DataCertification(
            compatibility_basis="axiom_oracle_local_policyengine_us_override",
            certified_for_model_version=runtime_model_version,
            data_build_fingerprint=runtime_data_build_fingerprint,
            certified_by="axiom-encode tax-ecps-compare",
        )

    manifest.certify_data_release_compatibility = _allow_local_oracle_data
    try:
        import policyengine.tax_benefit_models.common.model_version as model_version
    except ImportError:  # pragma: no cover - optional runtime dependency
        return
    model_version.certify_data_release_compatibility = _allow_local_oracle_data


def build_axiom_request(
    *,
    pe_data: dict[str, Any],
    year: int,
    surface: str = "ctc",
    oasdi_wage_base_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if surface == "ctc":
        return build_ctc_request(pe_data=pe_data, year=year)
    if surface == "standard-deduction":
        return build_standard_deduction_request(pe_data=pe_data, year=year)
    if surface == "capital-gain-definitions":
        return build_capital_gain_definitions_request(pe_data=pe_data, year=year)
    if surface == "eitc":
        return build_eitc_request(pe_data=pe_data, year=year)
    if surface in PAYROLL_SURFACES:
        return build_payroll_request(
            pe_data=pe_data,
            year=year,
            surface=surface,
            oasdi_wage_base_results=oasdi_wage_base_results,
        )
    raise ValueError(f"unsupported tax surface: {surface}")


def build_ctc_request(*, pe_data: dict[str, Any], year: int) -> dict[str, Any]:
    interval = {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }
    inputs: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    persons_by_tax_unit = group_person_rows_by_tax_unit(pe_data["persons"])

    for _idx, row in pe_data["tax_units"].iterrows():
        tax_unit_id = int(row["tax_unit_id"])
        entity_id = tax_entity_id(tax_unit_id)
        for name, value in project_tax_unit_inputs(row).items():
            inputs.append(
                input_record(f"{CTC_BASE}#input.{name}", entity_id, interval, value)
            )
        inputs.append(
            input_record(
                f"{CTC_H_BASE}#input.filing_status_is_joint_return",
                entity_id,
                interval,
                uses_joint_ctc_phaseout_threshold(str(row["filing_status"])),
            )
        )
        tax_unit_persons = persons_by_tax_unit.get(tax_unit_id, [])
        contexts = project_tax_unit_person_contexts(tax_unit_persons)
        for person_index, (person, context) in enumerate(
            zip(tax_unit_persons, contexts, strict=True)
        ):
            person_id = f"{entity_id}_person_{person_index}"
            relations.append(
                {
                    "name": f"{CTC_BASE}#relation.ctc_qualifying_child_of_tax_unit",
                    "tuple": [person_id, entity_id],
                    "interval": interval,
                }
            )
            relations.append(
                {
                    "name": f"{CTC_H_BASE}#relation.dependent_of_tax_unit",
                    "tuple": [person_id, entity_id],
                    "interval": interval,
                }
            )
            for name, value in project_ctc_person_inputs(person, context).items():
                inputs.append(
                    input_record(f"{CTC_BASE}#input.{name}", person_id, interval, value)
                )
            for name, value in project_ctc_h_person_inputs(person, context).items():
                inputs.append(
                    input_record(
                        f"{CTC_H_BASE}#input.{name}", person_id, interval, value
                    )
                )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": relations},
        "queries": [
            {
                "entity_id": tax_entity_id(tax_unit_id),
                "period": interval,
                "outputs": [spec["axiom"] for spec in CTC_OUTPUTS.values()],
            }
            for tax_unit_id in pe_data["tax_unit_ids"]
        ],
    }


def build_standard_deduction_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }
    inputs: list[dict[str, Any]] = []
    persons_by_tax_unit = group_person_rows_by_tax_unit(pe_data["persons"])
    for _idx, row in pe_data["tax_units"].iterrows():
        tax_unit_id = int(row["tax_unit_id"])
        entity_id = tax_entity_id(tax_unit_id)
        tax_unit_persons = persons_by_tax_unit.get(tax_unit_id, [])
        for name, value in project_standard_deduction_inputs(
            row=row,
            persons=tax_unit_persons,
        ).items():
            inputs.append(
                input_record(
                    f"{STANDARD_DEDUCTION_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": [
            {
                "entity_id": tax_entity_id(tax_unit_id),
                "period": interval,
                "outputs": [
                    spec["axiom"] for spec in STANDARD_DEDUCTION_OUTPUTS.values()
                ],
            }
            for tax_unit_id in pe_data["tax_unit_ids"]
        ],
    }


def build_capital_gain_definitions_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }
    inputs: list[dict[str, Any]] = []
    persons_by_tax_unit = group_person_rows_by_tax_unit(pe_data["persons"])
    for _idx, row in pe_data["tax_units"].iterrows():
        tax_unit_id = int(row["tax_unit_id"])
        entity_id = tax_entity_id(tax_unit_id)
        tax_unit_persons = persons_by_tax_unit.get(tax_unit_id, [])
        for name, value in project_capital_gain_definition_inputs(
            row=row,
            persons=tax_unit_persons,
        ).items():
            inputs.append(
                input_record(
                    f"{CAPITAL_GAINS_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": [
            {
                "entity_id": tax_entity_id(tax_unit_id),
                "period": interval,
                "outputs": [
                    spec["axiom"] for spec in CAPITAL_GAINS_DEFINITION_OUTPUTS.values()
                ],
            }
            for tax_unit_id in pe_data["tax_unit_ids"]
        ],
    }


def build_eitc_request(*, pe_data: dict[str, Any], year: int) -> dict[str, Any]:
    interval = {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }
    inputs: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    persons_by_tax_unit = group_person_rows_by_tax_unit(pe_data["persons"])

    for _idx, row in pe_data["tax_units"].iterrows():
        tax_unit_id = int(row["tax_unit_id"])
        entity_id = tax_entity_id(tax_unit_id)
        tax_unit_persons = persons_by_tax_unit.get(tax_unit_id, [])
        contexts = project_tax_unit_person_contexts(tax_unit_persons)
        for name, value in project_eitc_tax_unit_inputs(
            row=row,
            persons=tax_unit_persons,
        ).items():
            inputs.append(
                input_record(f"{EITC_BASE}#input.{name}", entity_id, interval, value)
            )
        for name, value in project_section_32_c_2_tax_unit_inputs(
            persons=tax_unit_persons,
            contexts=contexts,
        ).items():
            inputs.append(
                input_record(
                    f"{SECTION_32_C_2_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        for name, value in project_section_112_tax_unit_inputs().items():
            inputs.append(
                input_record(
                    f"{SECTION_112_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        for name, value in project_section_1402_a_tax_unit_inputs(
            persons=tax_unit_persons,
            contexts=contexts,
        ).items():
            inputs.append(
                input_record(
                    f"{SECTION_1402_A_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        for name, value in project_section_164_f_tax_unit_inputs().items():
            inputs.append(
                input_record(
                    f"{SECTION_164_F_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        for name, value in project_section_1401_tax_unit_inputs(
            row=row,
            persons=tax_unit_persons,
            contexts=contexts,
        ).items():
            inputs.append(
                input_record(
                    f"{SECTION_1401_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )

        for person_index, (person, context) in enumerate(
            zip(tax_unit_persons, contexts, strict=True)
        ):
            person_id = f"{entity_id}_person_{person_index}"
            relations.append(
                {
                    "name": f"{EITC_BASE}#relation.qualifying_child_of_tax_unit",
                    "tuple": [person_id, entity_id],
                    "interval": interval,
                }
            )
            for name, value in project_eitc_person_inputs(person, context).items():
                inputs.append(
                    input_record(
                        f"{EITC_BASE}#input.{name}", person_id, interval, value
                    )
                )
            for name, value in project_section_152_c_person_inputs(
                person,
                context,
            ).items():
                inputs.append(
                    input_record(
                        f"{SECTION_152_C_BASE}#input.{name}",
                        person_id,
                        interval,
                        value,
                    )
                )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": relations},
        "queries": [
            {
                "entity_id": tax_entity_id(tax_unit_id),
                "period": interval,
                "outputs": [spec["axiom"] for spec in EITC_OUTPUTS.values()],
            }
            for tax_unit_id in pe_data["tax_unit_ids"]
        ],
    }


def build_payroll_request(
    *,
    pe_data: dict[str, Any],
    year: int,
    surface: str,
    oasdi_wage_base_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = PAYROLL_SURFACES[surface]
    base = str(config["base"])
    wage_input = str(config["wage_input"])
    oasdi_taxable_wages = (
        taxable_oasdi_wages_by_person_id(
            pe_data=pe_data,
            oasdi_wage_base_results=oasdi_wage_base_results,
        )
        if wage_input == AXIOM_3121_TAXABLE_OASDI_WAGES
        else None
    )
    interval = {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }
    inputs: list[dict[str, Any]] = []
    for _idx, person in pe_data["persons"].iterrows():
        person_id = int(person["person_id"])
        wages = (
            oasdi_taxable_wages[person_id]
            if oasdi_taxable_wages is not None
            else money(person[wage_input])
        )
        inputs.append(
            input_record(
                f"{base}#input.wages",
                person_entity_id(person_id),
                interval,
                wages,
            )
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": [
            {
                "entity_id": person_entity_id(person_id),
                "period": interval,
                "outputs": [spec["axiom"] for spec in config["outputs"].values()],
            }
            for person_id in pe_data["person_ids"]
        ],
    }


def build_contribution_and_benefit_base_request(*, year: int) -> dict[str, Any]:
    interval = {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }
    return {
        "mode": "explain",
        "dataset": {"inputs": [], "relations": []},
        "queries": [
            {
                "entity_id": f"tax_year_{year}",
                "period": interval,
                "outputs": [contribution_and_benefit_base_output(year)],
            }
        ],
    }


def contribution_and_benefit_base_from_results(
    results: list[dict[str, Any]],
    *,
    year: int,
) -> float:
    output = contribution_and_benefit_base_output(year)
    if len(results) != 1:
        raise ValueError(
            "Expected exactly one contribution-and-benefit-base output result; "
            f"got {len(results)}."
        )
    outputs = results[0].get("outputs") or {}
    contribution_base = output_number(outputs.get(output))
    if math.isnan(contribution_base):
        raise ValueError(
            f"Axiom contribution-and-benefit-base output missing: {output}."
        )
    return contribution_base


def build_oasdi_wage_base_request(
    *,
    pe_data: dict[str, Any],
    year: int,
    contribution_base: float,
) -> dict[str, Any]:
    interval = {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }
    inputs: list[dict[str, Any]] = []
    for _idx, person in pe_data["persons"].iterrows():
        entity_id = person_entity_id(int(person["person_id"]))
        for name, value in project_oasdi_wage_base_inputs(
            person,
            contribution_base=contribution_base,
        ).items():
            inputs.append(
                input_record(
                    f"{OASDI_WAGE_BASE_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": [
            {
                "entity_id": person_entity_id(person_id),
                "period": interval,
                "outputs": [OASDI_WAGE_BASE_EXCLUSION_OUTPUT],
            }
            for person_id in pe_data["person_ids"]
        ],
    }


def project_oasdi_wage_base_inputs(
    person: Any,
    *,
    contribution_base: float,
) -> dict[str, Any]:
    return {
        (
            "successor_employer_acquired_substantially_all_trade_or_business_property_"
            "from_predecessor_during_calendar_year"
        ): False,
        (
            "successor_employer_acquired_substantially_all_property_used_in_separate_"
            "unit_of_predecessor_trade_or_business_during_calendar_year"
        ): False,
        "individual_employed_by_successor_immediately_after_acquisition": False,
        "individual_employed_by_predecessor_immediately_before_acquisition": False,
        (
            "predecessor_remuneration_other_than_succeeding_paragraphs_paid_or_"
            "considered_paid_to_individual_before_acquisition_during_calendar_year"
        ): 0,
        (
            "remuneration_other_than_succeeding_paragraphs_paid_to_individual_by_"
            "employer_during_calendar_year_before_payment"
        ): 0,
        "contribution_and_benefit_base_under_section_230_of_social_security_act": (
            contribution_base
        ),
        "remuneration_paid_to_individual_by_employer_with_respect_to_employment": (
            money(person["payroll_tax_gross_wages"])
        ),
    }


def taxable_oasdi_wages_by_person_id(
    *,
    pe_data: dict[str, Any],
    oasdi_wage_base_results: list[dict[str, Any]] | None,
) -> dict[int, float]:
    if oasdi_wage_base_results is None:
        raise ValueError("OASDI payroll surfaces require Axiom 3121 wage-base results.")
    person_ids = [int(person_id) for person_id in pe_data["person_ids"]]
    if len(oasdi_wage_base_results) != len(person_ids):
        raise ValueError(
            "Axiom 3121 wage-base result count does not match selected persons."
        )

    values: dict[int, float] = {}
    persons = pe_data["persons"].reset_index(drop=True)
    for index, person_id in enumerate(person_ids):
        outputs = oasdi_wage_base_results[index].get("outputs") or {}
        exclusion = output_number(outputs.get(OASDI_WAGE_BASE_EXCLUSION_OUTPUT))
        if math.isnan(exclusion):
            raise ValueError(
                f"Axiom 3121 wage-base output missing for {person_entity_id(person_id)}."
            )
        gross_wages = money(persons.iloc[index]["payroll_tax_gross_wages"])
        values[person_id] = max(0.0, gross_wages - exclusion)
    return values


def payroll_surface_uses_axiom_3121(surface: str) -> bool:
    config = PAYROLL_SURFACES.get(surface)
    return (
        isinstance(config, dict)
        and config.get("wage_input") == AXIOM_3121_TAXABLE_OASDI_WAGES
    )


def project_tax_unit_inputs(row: Any) -> dict[str, Any]:
    return {
        "adjusted_gross_income": money(row["adjusted_gross_income"]),
        "amount_excluded_from_gross_income_under_section_911": 0,
        "amount_excluded_from_gross_income_under_section_931": 0,
        "amount_excluded_from_gross_income_under_section_933": 0,
        "filing_status": filing_status_code(str(row["filing_status"])),
        "taxable_year_begins_after_2017": True,
        "taxpayer_identification_number_issued_after_return_due_date": False,
        "taxable_year_months": 12,
        "taxable_year_closed_by_reason_of_taxpayer_death": False,
        "ctc_fraud_disallowance_period_applies": False,
        "ctc_reckless_or_intentional_disregard_disallowance_period_applies": False,
        "prior_deficiency_denial_without_required_eligibility_information": False,
        "aggregate_advance_payments_under_section_7527A": 0,
    }


def project_standard_deduction_inputs(row: Any, persons: list[Any]) -> dict[str, Any]:
    filing_status = str(row["filing_status"])
    return {
        "filing_status": filing_status_code(filing_status),
        "may_be_claimed_as_dependent_by_another_taxpayer": False,
        "earned_income": 0,
        "additional_standard_deduction_entitlement_count_under_subsection_f": (
            additional_standard_deduction_entitlement_count(persons)
        ),
        "individual_is_unmarried_and_not_surviving_spouse": (
            individual_is_unmarried_and_not_surviving_spouse(filing_status)
        ),
    }


def project_capital_gain_definition_inputs(
    row: Any, persons: list[Any]
) -> dict[str, Any]:
    long_term_capital_gains = person_money_sum(
        persons,
        "long_term_capital_gains_before_response",
    )
    short_term_capital_gains = person_money_sum(persons, "short_term_capital_gains")
    qualified_dividend_income = person_money_sum(
        persons,
        "qualified_dividend_income",
    )
    investment_income_election = person_money_sum(
        persons,
        "investment_income_elected_form_4952",
    )
    capital_gains_28_percent_rate_gain = person_money_sum(
        persons,
        "long_term_capital_gains_on_collectibles",
    ) + person_money_sum(
        persons,
        "long_term_capital_gains_on_small_business_stock",
    )
    return {
        "long_term_capital_gains": long_term_capital_gains,
        "short_term_capital_gains": short_term_capital_gains,
        "qualified_dividend_income": qualified_dividend_income,
        "net_capital_gain_taken_into_account_as_investment_income_under_section_163_d_4_B_iii": (
            investment_income_election
        ),
        "unrecaptured_section_1250_gain": money(
            row.get("unrecaptured_section_1250_gain", 0)
        ),
        "capital_gains_28_percent_rate_gain": capital_gains_28_percent_rate_gain,
    }


def project_eitc_tax_unit_inputs(row: Any, persons: list[Any]) -> dict[str, Any]:
    filing_status = str(row["filing_status"])
    filing_status_numeric = filing_status_code(filing_status)
    contexts = project_tax_unit_person_contexts(persons)
    head_index, spouse_index = tax_unit_head_spouse_indices(persons)
    filer_indices = {index for index in (head_index, spouse_index) if index is not None}
    filer_contexts = [
        context for index, context in enumerate(contexts) if index in filer_indices
    ]
    spouse_has_required_ssn = spouse_index is not None and valid_child_ssn_type(
        str(persons[spouse_index].get("ssn_card_type", ""))
    )
    return {
        "filing_status": filing_status_numeric,
        "adjusted_gross_income": money(row["adjusted_gross_income"]),
        "eitc_relevant_investment_income": project_eitc_relevant_investment_income(
            row=row,
            persons=persons,
        ),
        "childless_taxpayer_principal_place_of_abode_in_united_states_more_than_half_year": True,
        "childless_taxpayer_or_spouse_age_eligible_for_eitc": any(
            context.is_head or context.is_spouse
            for context in filer_contexts
            if context.is_head or context.is_spouse
        )
        and any(25 <= money(persons[index]["age"]) < 65 for index in filer_indices),
        "taxpayer_is_dependent_for_section_151_to_another_taxpayer": False,
        "taxpayer_is_qualifying_child_of_another_taxpayer": False,
        "taxpayer_claims_section_911_benefits": False,
        "taxpayer_is_nonresident_alien_for_any_portion_of_year": False,
        "taxpayer_treated_as_resident_by_section_6013_g_or_h_election": False,
        "taxpayer_is_married_under_section_7703_a": filing_status_numeric in {1, 2},
        "satisfies_eitc_separated_spouse_rules": (
            filing_status_numeric == 2
            and any(bool_value(person.get("is_separated", False)) for person in persons)
        ),
        "taxable_year_is_full_12_months": True,
        "taxable_year_closed_by_reason_of_taxpayer_death": False,
        "eitc_disallowance_period_applies": False,
        "prior_deficiency_denial_without_required_eligibility_information": False,
        "taxpayer_includes_required_social_security_number_on_return": any(
            context.filer_has_valid_child_ctc_ssn for context in contexts
        ),
        "spouse_includes_required_social_security_number_on_return": (
            spouse_has_required_ssn
        ),
    }


def project_section_32_c_2_tax_unit_inputs(
    *,
    persons: list[Any],
    contexts: list[PersonProjectionContext],
) -> dict[str, Any]:
    return {
        "wages_salaries_tips_and_other_employee_compensation_includible_in_gross_income": sum(
            money(person.get("employment_income_before_lsr", 0))
            for person, context in zip(persons, contexts, strict=True)
            if context.is_head or context.is_spouse
        ),
        "pension_or_annuity_amounts_received": 0,
        "amounts_to_which_section_871_a_applies": 0,
        "amounts_received_for_services_while_inmate_at_penal_institution": 0,
        "subsidized_state_work_activity_amounts_received": 0,
        "taxpayer_elects_to_treat_section_112_excluded_amounts_as_earned_income": False,
    }


def project_section_112_tax_unit_inputs() -> dict[str, Any]:
    return {
        "member_below_grade_of_commissioned_officer_in_armed_forces": False,
        "served_in_combat_zone_during_month": False,
        "hospitalized_resulting_from_combat_zone_wounds_disease_or_injury": False,
        "months_beginning_after_combatant_activities_termination": 0,
        "vietnam_combat_zone_hospitalization_month_after_january_1978": False,
        "active_service_compensation_as_enlisted_member_excluding_pensions_and_retirement_pay": 0,
        "commissioned_officer_in_armed_forces_excluding_commissioned_warrant_officer": False,
        "active_service_compensation_as_commissioned_officer_excluding_pensions_and_retirement_pay": 0,
        "maximum_enlisted_amount_for_commissioned_officer_months": 0,
        "armed_forces_member_in_missing_status_during_vietnam_conflict_as_result_of_conflict": False,
        "officially_absent_from_post_of_duty_without_authority": False,
        "armed_forces_missing_status_active_service_compensation": 0,
        "civilian_employee_in_missing_status_during_vietnam_conflict_as_result_of_conflict": False,
        "civilian_employee_missing_status_active_service_compensation": 0,
    }


def project_section_1402_a_tax_unit_inputs(
    *,
    persons: list[Any],
    contexts: list[PersonProjectionContext],
) -> dict[str, Any]:
    self_employment_income = sum(
        money(person.get("self_employment_income_before_lsr", 0))
        + money(person.get("sstb_self_employment_income_before_lsr", 0))
        for person, context in zip(persons, contexts, strict=True)
        if context.is_head or context.is_spouse
    )
    return {
        "self_employment_trade_or_business_gross_income": self_employment_income,
        "self_employment_trade_or_business_deductions": 0,
        "partnership_section_702_a_8_income_or_loss": 0,
    }


def project_section_164_f_tax_unit_inputs() -> dict[str, Any]:
    return {"taxpayer_is_individual": True}


def project_section_1401_tax_unit_inputs(
    *,
    row: Any,
    persons: list[Any],
    contexts: list[PersonProjectionContext],
) -> dict[str, Any]:
    section_1402_inputs = project_section_1402_a_tax_unit_inputs(
        persons=persons,
        contexts=contexts,
    )
    net_earnings_before_paragraph_12 = (
        section_1402_inputs["self_employment_trade_or_business_gross_income"]
        - section_1402_inputs["self_employment_trade_or_business_deductions"]
        + section_1402_inputs["partnership_section_702_a_8_income_or_loss"]
    )
    net_earnings_after_paragraph_12 = net_earnings_before_paragraph_12 * (1 - 0.0765)
    self_employment_income = (
        0
        if net_earnings_after_paragraph_12 < 400
        else max(0, net_earnings_after_paragraph_12)
    )
    return {
        "international_social_security_agreement_under_section_233_in_effect": False,
        "filing_status": filing_status_code(str(row["filing_status"])),
        "self_employment_income": money(self_employment_income),
        "wages_taken_into_account_for_additional_medicare_tax": 0,
    }


def project_eitc_relevant_investment_income(row: Any, persons: list[Any]) -> float:
    net_capital_gains = person_money_sum(
        persons,
        "long_term_capital_gains_before_response",
    ) + person_money_sum(persons, "short_term_capital_gains")
    return (
        person_money_sum(persons, "taxable_interest_income")
        + person_money_sum(persons, "tax_exempt_interest_income")
        + person_money_sum(persons, "qualified_dividend_income")
        + person_money_sum(persons, "non_qualified_dividend_income")
        + person_money_sum(persons, "rental_income")
        + max(0.0, net_capital_gains)
    )


def project_eitc_person_inputs(
    person: Any,
    context: PersonProjectionContext,
) -> dict[str, Any]:
    return {
        "qualifying_child_principal_place_of_abode_is_in_united_states": True,
        "qualifying_child_name_age_and_tin_included_on_return": (
            context.has_valid_child_ssn
        ),
        "qualifying_child_is_married_at_close_of_taxable_year": False,
        "taxpayer_entitled_to_section_151_deduction_for_child_or_would_be_but_for_section_152_e": (
            context.is_tax_unit_dependent
        ),
    }


def project_section_152_c_person_inputs(
    person: Any,
    context: PersonProjectionContext,
) -> dict[str, Any]:
    return {
        "individual_is_child_of_taxpayer_or_descendant_of_such_child": (
            context.is_tax_unit_dependent
        ),
        "individual_is_sibling_stepsibling_or_descendant_of_such_relative": False,
        "individual_principal_place_of_abode_with_taxpayer_fraction": (
            1.0 if context.is_tax_unit_dependent else 0.0
        ),
        "individual_is_permanently_and_totally_disabled": bool_value(
            person.get("is_permanently_and_totally_disabled", False)
        ),
        "individual_is_younger_than_taxpayer": context.is_tax_unit_dependent,
        "individual_age_at_close_of_calendar_year": money(person["age"]),
        "individual_is_student": bool_value(
            person.get("is_full_time_college_student", False)
        ),
        "individual_filed_joint_return_with_spouse_other_than_only_for_claim_of_refund": False,
        "individual_may_be_claimed_as_qualifying_child_by_two_or_more_taxpayers": False,
        "parents_of_individual_may_claim_individual_but_no_parent_claims": False,
        "taxpayer_is_parent_of_individual": context.is_tax_unit_dependent,
        "taxpayer_adjusted_gross_income_higher_than_highest_parent_adjusted_gross_income": False,
        "parents_claiming_child_do_not_file_joint_return_together": False,
        "child_resided_with_taxpayer_parent_for_longest_period": False,
        "child_resided_with_both_parents_same_amount_of_time_and_taxpayer_parent_has_highest_adjusted_gross_income": False,
        "no_parent_of_individual_is_a_claiming_taxpayer": False,
        "taxpayer_has_highest_adjusted_gross_income_among_claiming_taxpayers": False,
    }


def additional_standard_deduction_entitlement_count(persons: list[Any]) -> int:
    head_index, spouse_index = tax_unit_head_spouse_indices(persons)
    count = 0
    for index in {head_index, spouse_index}:
        if index is None:
            continue
        person = persons[index]
        if money(person["age"]) >= 65:
            count += 1
        if bool_value(person.get("is_blind", False)):
            count += 1
    return count


def individual_is_unmarried_and_not_surviving_spouse(value: str) -> bool:
    return value.strip().upper() in {"SINGLE", "HEAD_OF_HOUSEHOLD"}


def project_ctc_person_inputs(
    person: Any, context: PersonProjectionContext | None = None
) -> dict[str, Any]:
    context = context or project_tax_unit_person_contexts([person])[0]
    age = money(person["age"])
    return {
        "age": age,
        "qualifying_child_under_section_152_c": (
            context.qualifying_child_under_section_152_c
        ),
        "allowed_deduction_under_section_151_for_child": (
            context.is_tax_unit_dependent
        ),
        "certain_noncitizen_exception_applies": False,
        "ctc_child_missing_identification": bool(
            context.qualifying_child_described_in_subsection_c
            and not context.has_valid_child_ssn
        ),
        "qualifying_child_name_included_on_return": context.has_valid_child_ssn,
        "qualifying_child_tin_included_on_return": context.has_valid_child_ssn,
        "qualifying_child_tin_issued_on_or_before_return_due_date": (
            context.has_valid_child_ssn
        ),
    }


def project_ctc_h_person_inputs(
    person: Any, context: PersonProjectionContext | None = None
) -> dict[str, Any]:
    context = context or project_tax_unit_person_contexts([person])[0]
    return {
        "dependent_under_section_152": context.is_tax_unit_dependent,
        "qualifying_child_described_in_subsection_c": (
            context.qualifying_child_described_in_subsection_c
        ),
        "noncitizen_exception_to_other_dependent_credit_under_subsection_h": False,
        "taxpayer_or_spouse_ssn_included_on_return": (
            context.filer_has_valid_child_ctc_ssn
        ),
        "qualifying_child_ssn_included_on_return": context.has_valid_child_ssn,
        "taxpayer_or_spouse_ssn_is_valid_for_subsection_h": (
            context.filer_has_valid_child_ctc_ssn
        ),
        "qualifying_child_ssn_is_valid_for_subsection_h": context.has_valid_child_ssn,
    }


def project_tax_unit_person_contexts(
    persons: list[Any],
) -> list[PersonProjectionContext]:
    head_index, spouse_index = tax_unit_head_spouse_indices(persons)
    filer_has_valid_child_ctc_ssn = any(
        index in {head_index, spouse_index}
        and valid_child_ssn_type(str(person.get("ssn_card_type", "")))
        for index, person in enumerate(persons)
    )
    contexts: list[PersonProjectionContext] = []
    for index, person in enumerate(persons):
        age = money(person["age"])
        is_head = index == head_index
        is_spouse = index == spouse_index
        is_tax_unit_dependent = not is_head and not is_spouse
        qualifying_child_under_section_152_c = is_tax_unit_dependent and (
            age < 19
            or (
                bool_value(person.get("is_full_time_college_student", False))
                and age < 24
            )
            or bool_value(person.get("is_permanently_and_totally_disabled", False))
        )
        ctc_qualifying_child = (
            qualifying_child_under_section_152_c
            and age < 17
            and not bool_value(
                person.get("certain_noncitizen_exception_applies", False)
            )
        )
        contexts.append(
            PersonProjectionContext(
                is_head=is_head,
                is_spouse=is_spouse,
                is_tax_unit_dependent=is_tax_unit_dependent,
                qualifying_child_under_section_152_c=(
                    qualifying_child_under_section_152_c
                ),
                qualifying_child_described_in_subsection_c=ctc_qualifying_child,
                has_valid_child_ssn=valid_child_ssn_type(
                    str(person.get("ssn_card_type", ""))
                ),
                filer_has_valid_child_ctc_ssn=filer_has_valid_child_ctc_ssn,
            )
        )
    return contexts


def tax_unit_head_spouse_indices(persons: list[Any]) -> tuple[int | None, int | None]:
    adult_indices = [
        index for index, person in enumerate(persons) if money(person["age"]) >= 18
    ]
    if not adult_indices:
        return None, None
    likely_adult_student_dependent_indices = {
        index
        for index in adult_indices
        if money(persons[index]["age"]) < 24
        and bool_value(persons[index].get("is_full_time_college_student", False))
        and any(
            other_index != index and money(persons[other_index]["age"]) >= 25
            for other_index in adult_indices
        )
    }
    filer_adult_indices = [
        index
        for index in adult_indices
        if index not in likely_adult_student_dependent_indices
    ] or adult_indices
    head_index = max(
        filer_adult_indices,
        key=lambda index: (
            bool_value(persons[index].get("is_household_head", False)),
            money(persons[index]["age"]),
        ),
    )
    separated = any(bool_value(person.get("is_separated", False)) for person in persons)
    if separated:
        return head_index, None
    spouse_candidates = [index for index in filer_adult_indices if index != head_index]
    spouse_index = (
        max(spouse_candidates, key=lambda index: money(persons[index]["age"]))
        if spouse_candidates
        else None
    )
    return head_index, spouse_index


def run_axiom_program(
    *,
    program: Path,
    request: dict[str, Any],
    rulespec_root: Path,
    axiom_rules_path: Path,
) -> list[dict[str, Any]]:
    binary = axiom_rules_path / "target" / "debug" / "axiom-rules-engine"
    if not binary.exists():
        raise SystemExit(f"axiom-rules-engine binary not found: {binary}")
    env = os.environ.copy()
    env["AXIOM_RULESPEC_REPO_ROOTS"] = str(rulespec_root)
    with tempfile.TemporaryDirectory(prefix="axiom-tax-ecps-") as tmpdir:
        artifact = Path(tmpdir) / f"{program.stem}.json"
        compile_result = subprocess.run(
            [
                str(binary),
                "compile",
                "--program",
                str(program),
                "--output",
                str(artifact),
            ],
            capture_output=True,
            text=True,
            cwd=str(axiom_rules_path),
            env=env,
            timeout=60,
        )
        if compile_result.returncode != 0:
            raise SystemExit(
                compile_result.stderr.strip() or compile_result.stdout.strip()
            )
        run_result = subprocess.run(
            [str(binary), "run-compiled", "--artifact", str(artifact)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            cwd=str(axiom_rules_path),
            env=env,
            timeout=120,
        )
        if run_result.returncode != 0:
            raise SystemExit(run_result.stderr.strip() or run_result.stdout.strip())
        return json.loads(run_result.stdout)["results"]


def compare_outputs(
    *,
    pe_data: dict[str, Any],
    axiom_outputs_by_surface: dict[str, list[dict[str, Any]]],
    tolerance: float,
    relative_tolerance: float,
) -> TaxComparisonReport:
    mismatches: list[TaxComparisonRow] = []
    summary: dict[str, dict[str, Any]] = {
        f"{surface}:{name}": {
            "surface": surface,
            "output": name,
            "compared": 0,
            "mismatches": 0,
            "max_abs_diff": 0.0,
            "max_relative_diff": 0.0,
        }
        for surface, outputs in SURFACE_OUTPUTS.items()
        if surface in axiom_outputs_by_surface
        for name in outputs
    }
    tax_units = pe_data["tax_units"].reset_index(drop=True)
    persons = pe_data["persons"].reset_index(drop=True)
    compared_values = 0
    for surface, axiom_outputs in axiom_outputs_by_surface.items():
        output_specs = SURFACE_OUTPUTS[surface]
        source_rows = persons if surface in PAYROLL_SURFACES else tax_units
        source_ids = (
            pe_data["person_ids"]
            if surface in PAYROLL_SURFACES
            else pe_data["tax_unit_ids"]
        )
        for index, result in enumerate(axiom_outputs):
            entity_id = (
                person_entity_id(int(source_ids[index]))
                if surface in PAYROLL_SURFACES
                else tax_entity_id(int(source_ids[index]))
            )
            outputs = result.get("outputs") or {}
            pe_row = source_rows.iloc[index]
            for name, spec in output_specs.items():
                axiom_value = output_number(outputs.get(spec["axiom"]))
                pe_value = money(pe_row[spec["pe"]])
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
                    summary[summary_key]["mismatches"] += 1
                    mismatches.append(
                        TaxComparisonRow(
                            surface=surface,
                            entity_id=entity_id,
                            output=name,
                            axiom=axiom_value,
                            policyengine=pe_value,
                            diff=diff,
                        )
                    )
    return TaxComparisonReport(
        compared_tax_units=len(pe_data["tax_unit_ids"]),
        compared_persons=len(pe_data["person_ids"]),
        compared_values=compared_values,
        mismatches=mismatches,
        output_summary=list(summary.values()),
        projection_notes=[
            "Current CTC projection uses ECPS raw tax-unit membership, age, "
            "student status, separation status, and SSN-card type to reconstruct "
            "the structural Section 152 facts needed by 26 USC 24.",
            "Standard deduction projection uses ECPS raw ages, blindness, and "
            "tax-unit membership to reconstruct aged-or-blind counts under "
            "26 USC 63(f).",
            "AGI and filing status remain boundary inputs until upstream AGI, "
            "filing-status, and return-filing rules are encoded end-to-end.",
            "The standard deduction comparison currently treats dependency by "
            "another taxpayer and earned income as boundary-false/zero because "
            "those upstream facts are not yet encoded from ECPS leaf inputs.",
            "OASDI payroll projections run Axiom 3121(a)(1) before sections "
            "3101(a) and 3111(a), so taxable OASDI wages are no longer taken "
            "from PolicyEngine. The section 230 contribution-and-benefit base "
            "is resolved by running the encoded SSA automatic-determination "
            "RuleSpec before section 3121(a)(1).",
            "Capital-gain-definition projections compare Section 1(h) net and "
            "adjusted net capital gain from ECPS raw capital-gain, dividend, "
            "investment-income-election, and unrecaptured-section-1250 fields. "
            "The full capital-gains-tax surface waits on encoded upstream "
            "taxable income rather than using PolicyEngine taxable income as "
            "an Axiom input.",
            "EITC projections supply ECPS wage and self-employment leaf facts "
            "to Sections 32(c)(2) and 1402(a); Section 32 earned income is "
            "computed by Axiom rather than passed in as a PolicyEngine output.",
        ],
    )


def print_report(
    report: TaxComparisonReport, *, tolerance: float, relative_tolerance: float
) -> None:
    print("PolicyEngine tax ECPS comparison")
    print(f"Compared tax units: {report.compared_tax_units:,}")
    print(f"Compared persons: {report.compared_persons:,}")
    print(f"Compared values: {report.compared_values:,}")
    print(f"Tolerance: {tolerance:g}")
    print(f"Relative tolerance: {relative_tolerance:g}")
    print(f"Mismatches: {len(report.mismatches):,}")
    print()
    print("By output:")
    for item in report.output_summary:
        print(
            f"  - {item['surface']}:{item['output']}: "
            f"{item['mismatches']:,}/{item['compared']:,} mismatch, "
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
                f"axiom={row.axiom:.2f} pe={row.policyengine:.2f} diff={row.diff:.2f}"
            )
    print()
    print("Projection notes:")
    for note in report.projection_notes:
        print(f"  - {note}")


def group_person_rows_by_tax_unit(persons: Any) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {}
    for _idx, person in persons.iterrows():
        grouped.setdefault(int(person["person_tax_unit_id"]), []).append(person)
    return grouped


def person_money_sum(persons: list[Any], column: str) -> float:
    return sum(money(person.get(column, 0)) for person in persons)


def input_record(
    name: str, entity_id: str, interval: dict[str, str], value: Any
) -> dict[str, Any]:
    return {
        "name": name,
        "entity": "Entity",
        "entity_id": entity_id,
        "interval": interval,
        "value": scalar_value(value),
    }


def scalar_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, float):
        return {"kind": "decimal", "value": str(value)}
    return {"kind": "text", "value": str(value)}


def output_number(output: Any) -> float:
    if not isinstance(output, dict):
        return math.nan
    if output.get("kind") == "judgment":
        outcome = str(output.get("outcome", "")).strip().lower()
        if outcome == "holds":
            return 1.0
        if outcome == "not_holds":
            return 0.0
    value = output.get("value") or {}
    raw = value.get("value")
    if raw is None:
        return math.nan
    if isinstance(raw, bool):
        return float(raw)
    return float(raw)


def within_tolerance(
    axiom_value: float,
    policyengine_value: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    return math.isclose(
        axiom_value,
        policyengine_value,
        abs_tol=absolute_tolerance,
        rel_tol=relative_tolerance,
    )


def relative_diff(axiom_value: float, policyengine_value: float) -> float:
    denominator = max(abs(axiom_value), abs(policyengine_value))
    if denominator == 0:
        return 0.0
    return abs(axiom_value - policyengine_value) / denominator


def filing_status_code(value: str) -> int:
    normalized = value.strip().upper()
    if normalized not in FILING_STATUS_CODES:
        raise ValueError(f"unsupported filing status: {value}")
    return FILING_STATUS_CODES[normalized]


def uses_joint_ctc_phaseout_threshold(value: str) -> bool:
    return value.strip().upper() in {"JOINT", "SURVIVING_SPOUSE"}


def valid_child_ssn_type(value: str) -> bool:
    return value.strip().upper() in VALID_CHILD_SSN_TYPES


def tax_entity_id(tax_unit_id: int) -> str:
    return f"tax_unit_{tax_unit_id}"


def person_entity_id(person_id: int) -> str:
    return f"person_{person_id}"


def money(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if np is not None and np.isnan(value):
            return 0.0
    except TypeError:
        pass
    return float(value)


def bool_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if np is not None and np.isnan(value):
            return False
    except TypeError:
        pass
    return bool(value)


def resolve_workspace_root(root: Path | None) -> Path:
    if root is not None:
        return root.resolve()
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents, Path.home() / "TheAxiomFoundation"]:
        if (candidate / "rulespec-us").exists() and (
            candidate / "axiom-rules-engine"
        ).exists():
            return candidate
    return cwd


def require_numpy() -> None:
    if np is None:
        raise SystemExit(policyengine_install_message())


def require_policyengine_versions(
    *, allow_policyengine_us_version: bool = False
) -> None:
    try:
        policyengine_version = version("policyengine")
        policyengine_us_version = version("policyengine-us")
    except PackageNotFoundError as exc:
        raise SystemExit(policyengine_install_message()) from exc
    if policyengine_version != POLICYENGINE_VERSION:
        raise SystemExit(
            f"policyengine=={POLICYENGINE_VERSION} required; found "
            f"{policyengine_version}. {policyengine_install_message()}"
        )
    if (
        policyengine_us_version != POLICYENGINE_US_VERSION
        and not allow_policyengine_us_version
    ):
        raise SystemExit(
            f"policyengine-us=={POLICYENGINE_US_VERSION} required; found "
            f"{policyengine_us_version}. {policyengine_install_message()}"
        )


def policyengine_install_message() -> str:
    return (
        "Run with: uv run --with policyengine==4.4.4 "
        "--with policyengine-us==1.691.3 axiom-encode tax-ecps-compare"
    )


def log(message: str) -> None:
    print(message, file=sys.stderr)

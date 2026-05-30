"""Compare UK RuleSpec output against PolicyEngine Enhanced FRS.

The UK counterpart to the ECPS tax comparators starts with surfaces whose
RuleSpec inputs can be projected from PolicyEngine's Enhanced FRS without using
the compared output itself as an input. Surfaces that are mapped in the oracle
registry but still require legal predicates not exposed by the EFRS projection
are reported as skipped rather than silently ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ecps_tax import (
    POLICYENGINE_VERSION,
    input_record,
    money,
    output_number,
    relative_diff,
    run_axiom_program,
    within_tolerance,
)

DATASET = "hf://policyengine/policyengine-uk-data/enhanced_frs_2023_24.h5"
DATASET_STEM = "enhanced_frs_2023_24"

PERSONAL_ALLOWANCE_PROGRAM_PATH = Path("statutes/ukpga/2007/3/35.yaml")
PERSONAL_ALLOWANCE_BASE = "uk:statutes/ukpga/2007/3/35"

PERSONAL_ALLOWANCE_OUTPUTS = {
    "personal_allowance": {
        "axiom": f"{PERSONAL_ALLOWANCE_BASE}#personal_allowance",
        "pe": "personal_allowance",
    },
}


@dataclass(frozen=True)
class UKEFRSSurfaceSpec:
    program: Path
    entity: str
    outputs: dict[str, dict[str, str]]
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
}

SKIPPED_SURFACES = [
    {
        "surface": "child-benefit",
        "reason": (
            "rulespec-uk currently exposes the branch-selected weekly rate, but "
            "the generated rule depends on eldest-child, relationship "
            "coordination, voluntary-organisation, and specified-benefit legal "
            "predicates that are not yet projected from EFRS rows."
        ),
    },
    {
        "surface": "pension-credit",
        "reason": (
            "rulespec-uk currently exposes statutory amount components and "
            "branch controls for the standard minimum guarantee, severe "
            "disability addition, and carer addition; the EFRS harness still "
            "needs a benunit-level legal-predicate projection before those can "
            "be compared row by row."
        ),
    },
    {
        "surface": "universal-credit",
        "reason": (
            "rulespec-uk currently encodes table amounts for Regulation 36. "
            "Those are parameter checks rather than EFRS row-level benefit "
            "comparisons until the generated RuleSpec includes the composed "
            "Universal Credit award surface."
        ),
    },
]


@dataclass(frozen=True)
class UKEFRSComparisonRow:
    surface: str
    entity_id: str
    output: str
    axiom: float
    policyengine: float
    diff: float


@dataclass(frozen=True)
class UKEFRSComparisonReport:
    compared_persons: int
    compared_values: int
    mismatches: list[UKEFRSComparisonRow]
    output_summary: list[dict[str, Any]]
    skipped_surfaces: list[dict[str, str]]
    projection_notes: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "compared_persons": self.compared_persons,
            "compared_values": self.compared_values,
            "mismatch_count": len(self.mismatches),
            "mismatches": [row.__dict__ for row in self.mismatches],
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
        data_folder=data_folder,
        person_ids=person_ids,
        person_variables=policyengine_person_variables_for_surfaces(surfaces),
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


def load_policyengine_uk_data(
    *,
    year: int,
    sample_size: int,
    data_folder: Path,
    person_ids: tuple[int, ...] = (),
    person_variables: tuple[str, ...] = SURFACE_SPECS[
        "personal-allowance"
    ].pe_variables,
) -> dict[str, Any]:
    try:
        from policyengine.core import Simulation
        from policyengine.tax_benefit_models.uk import ensure_datasets, uk_latest
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc

    log("Loading PolicyEngine UK EFRS...")
    datasets = ensure_datasets(
        datasets=[DATASET],
        years=[year],
        data_folder=str(data_folder),
    )
    dataset = datasets[f"{DATASET_STEM}_{year}"]
    sim = Simulation(
        dataset=dataset,
        tax_benefit_model_version=uk_latest,
        extra_variables={"person": list(person_variables)},
    )
    log("Running PolicyEngine UK outputs...")
    sim.run()

    raw_persons = dataset.data.person[["person_id", "person_weight"]].copy()
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
    return {
        "persons": selected,
        "person_ids": [int(row_value(row, "person_id")) for row in selected],
    }


def policyengine_person_variables_for_surfaces(
    surfaces: list[str],
) -> tuple[str, ...]:
    variables: set[str] = set()
    for surface in surfaces:
        variables.update(SURFACE_SPECS[surface].pe_variables)
    return tuple(sorted(variables))


def select_person_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    person_ids: tuple[int, ...] = (),
) -> list[int]:
    eligible = [
        index
        for index, row in enumerate(rows)
        if money(row_value(row, "person_weight", 0)) > 0
    ]
    if not person_ids:
        return eligible if sample_size <= 0 else eligible[:sample_size]

    requested_ids = tuple(dict.fromkeys(int(value) for value in person_ids))
    index_by_id = {
        int(row_value(row, "person_id")): index for index, row in enumerate(rows)
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
            "Requested EFRS person_id not found: "
            + ", ".join(str(value) for value in missing)
        )
    if filtered:
        raise SystemExit(
            "Requested EFRS person_id is not eligible for this comparison: "
            + ", ".join(str(value) for value in filtered)
        )
    return selected


def build_axiom_request(
    *,
    pe_data: dict[str, Any],
    year: int,
    surface: str = "personal-allowance",
) -> dict[str, Any]:
    if surface == "personal-allowance":
        return build_personal_allowance_request(pe_data=pe_data, year=year)
    raise ValueError(f"unsupported UK EFRS surface: {surface}")


def build_personal_allowance_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = tax_year_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in pe_data["persons"]:
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


def project_personal_allowance_inputs(row: Any) -> dict[str, Any]:
    adjusted_net_income = money(row_value(row, "adjusted_net_income"))
    gift_aid_grossed_up = money(row_value(row, "gift_aid_grossed_up", 0))
    return {
        "individual_makes_claim": True,
        "individual_meets_requirements_under_section_56": True,
        "adjusted_net_income": max(0.0, adjusted_net_income - gift_aid_grossed_up),
    }


def compare_outputs(
    *,
    pe_data: dict[str, Any],
    axiom_outputs_by_surface: dict[str, list[dict[str, Any]]],
    tolerance: float,
    relative_tolerance: float,
) -> UKEFRSComparisonReport:
    mismatches: list[UKEFRSComparisonRow] = []
    summary: dict[str, dict[str, Any]] = {
        f"{surface}:{name}": {
            "surface": surface,
            "output": name,
            "compared": 0,
            "mismatches": 0,
            "max_abs_diff": 0.0,
            "max_relative_diff": 0.0,
        }
        for surface, spec in SURFACE_SPECS.items()
        if surface in axiom_outputs_by_surface
        for name in spec.outputs
    }
    compared_values = 0
    persons = pe_data["persons"]
    for surface, axiom_outputs in axiom_outputs_by_surface.items():
        output_specs = SURFACE_SPECS[surface].outputs
        for index, result in enumerate(axiom_outputs):
            pe_row = persons[index]
            entity_id = person_entity_id(int(row_value(pe_row, "person_id")))
            outputs = result.get("outputs") or {}
            for name, spec in output_specs.items():
                axiom_value = output_number(outputs.get(spec["axiom"]))
                pe_value = money(row_value(pe_row, spec["pe"]))
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
        compared_values=compared_values,
        mismatches=mismatches,
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
        ],
    )


def print_report(
    report: UKEFRSComparisonReport,
    *,
    tolerance: float,
    relative_tolerance: float,
) -> None:
    print("PolicyEngine UK EFRS comparison")
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
                f"axiom={row.axiom:.2f} pe={row.policyengine:.2f} "
                f"diff={row.diff:.2f}"
            )
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


def person_entity_id(person_id: int) -> str:
    return f"person_{person_id}"


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

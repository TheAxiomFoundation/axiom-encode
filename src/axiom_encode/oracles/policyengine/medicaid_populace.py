"""Compare Medicaid RuleSpec eligibility against PolicyEngine over Populace.

This command runs the real Axiom rules engine over Populace-derived person
records.  Its initial projection maps PolicyEngine's Medicaid component
predicates into the corresponding RuleSpec source inputs so residual
differences identify surface-composition gaps, not missing microdata plumbing.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom_encode.oracles.policyengine.ecps_snap import (
    Period,
    axiom_rules_env,
    compile_program,
    load_base_inputs,
    month_period,
    output_to_python,
    outputs_by_reference,
    resolve_axiom_binary,
    resolve_test_template_path,
    resolve_workspace_root,
    scalar_value,
)
from axiom_encode.oracles.policyengine.population import (
    DEFAULT_US_POPULACE_YEAR,
    load_populace_dataset,
    populace_data_requirement,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without optional oracle deps
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only without optional oracle deps
    pd = None


COMPARED_POLICYENGINE_OUTPUT = "is_medicaid_eligible"
COMPARED_AXIOM_OUTPUT_ID = "us:statutes/42/1396a/a/10#is_medicaid_eligible"
AXIOM_COMPONENT_OUTPUT_IDS = {
    "parent": "us:regulations/42-cfr/435/110#parent_or_caretaker_relative_eligible",
    "pregnant": "us:regulations/42-cfr/435/116#pregnant_woman_eligible",
    "child": "us:regulations/42-cfr/435/118#infants_and_children_eligible",
    "adult": "us:regulations/42-cfr/435/119#adult_group_eligible",
    "ssi": (
        "us:regulations/42-cfr/435/120/ssi-mandatory-group"
        "#medicaid_required_for_ssi_mandatory_group"
    ),
    "former_foster": (
        "us:regulations/42-cfr/435/150#former_foster_care_child_medicaid_required"
    ),
    "community_engagement": (
        "us:statutes/42/1396a/xx#demonstrated_community_engagement_for_month"
    ),
}
DEFAULT_PROGRAM_RELATIVE_PATH = Path("us/statutes/42/1396a/a/10.yaml")
STATE_FIPS_BY_CODE = {
    "AL": 1,
    "AK": 2,
    "AZ": 4,
    "AR": 5,
    "CA": 6,
    "CO": 8,
    "CT": 9,
    "DE": 10,
    "DC": 11,
    "FL": 12,
    "GA": 13,
    "HI": 15,
    "ID": 16,
    "IL": 17,
    "IN": 18,
    "IA": 19,
    "KS": 20,
    "KY": 21,
    "LA": 22,
    "ME": 23,
    "MD": 24,
    "MA": 25,
    "MI": 26,
    "MN": 27,
    "MS": 28,
    "MO": 29,
    "MT": 30,
    "NE": 31,
    "NV": 32,
    "NH": 33,
    "NJ": 34,
    "NM": 35,
    "NY": 36,
    "NC": 37,
    "ND": 38,
    "OH": 39,
    "OK": 40,
    "OR": 41,
    "PA": 42,
    "RI": 44,
    "SC": 45,
    "SD": 46,
    "TN": 47,
    "TX": 48,
    "UT": 49,
    "VT": 50,
    "VA": 51,
    "WA": 53,
    "WV": 54,
    "WI": 55,
    "WY": 56,
}


@dataclass
class PersonCase:
    person_index: int
    state: str
    inputs: dict[str, Any]
    pe_outputs: dict[str, Any]


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--year", type=int, default=2027)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument(
        "--state",
        default=None,
        help="Optional two-letter state code filter.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit after filtering. Omit to run all matching Populace people.",
    )
    parser.add_argument(
        "--pre-sample-households",
        type=int,
        default=None,
        help=(
            "Subset the local Populace household sample before constructing "
            "PolicyEngine. This preserves whole households and avoids building "
            "the full Medicaid graph for smoke tests."
        ),
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Only compare people PolicyEngine marks Medicaid-eligible.",
    )
    parser.add_argument(
        "--projection",
        choices=("policyengine-components",),
        default="policyengine-components",
        help=(
            "Projection mode. policyengine-components maps PE category "
            "component predicates into RuleSpec source inputs to isolate final "
            "surface-composition differences."
        ),
    )
    parser.add_argument(
        "--populace-year",
        type=int,
        default=DEFAULT_US_POPULACE_YEAR,
        help="Published US Populace dataset year to load.",
    )
    parser.add_argument(
        "--program",
        type=Path,
        default=None,
        help="Override the Medicaid RuleSpec program file.",
    )
    parser.add_argument(
        "--test-template",
        type=Path,
        default=None,
        help="Override the companion test template used for base input defaults.",
    )
    parser.add_argument(
        "--axiom-binary",
        type=Path,
        default=None,
        help="Path to the axiom-rules-engine binary.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Workspace containing axiom-rules-engine and rulespec-us.",
    )
    parser.add_argument("--write-csv", type=Path, default=None)
    parser.add_argument("--max-differences", type=int, default=20)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero when any compared row mismatches.",
    )
    parser.add_argument(
        "--min-match-rate",
        type=float,
        default=None,
        help="Exit nonzero when the match rate is below this threshold.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    return parser.parse_args()


def require_numpy() -> None:
    if np is None:
        raise SystemExit(
            "numpy is required. Run with: "
            "uv run --with numpy --with policyengine-core==3.26.0 --with "
            f"{populace_data_requirement('us')} axiom-encode "
            "medicaid-populace-compare"
        )


def require_pandas() -> None:
    if pd is None:
        raise SystemExit(
            "pandas is required. Run with: "
            "uv run --with pandas --with numpy --with policyengine-core==3.26.0 "
            f"--with {populace_data_requirement('us')} axiom-encode "
            "medicaid-populace-compare"
        )


def array(values: Any) -> np.ndarray:
    require_numpy()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    if hasattr(values, "values"):
        return np.asarray(values.values)
    return np.asarray(values)


def calculate(sim: Any, name: str, period: str | int) -> np.ndarray:
    return array(sim.calculate(name, period=period))


def calculate_or_default(
    sim: Any, name: str, period: str | int, default: Any, size: int
) -> np.ndarray:
    try:
        return calculate(sim, name, period)
    except Exception:
        return np.full(size, default)


def resolve_program_path(workspace_root: Path, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    cwd_program = Path.cwd() / DEFAULT_PROGRAM_RELATIVE_PATH
    if cwd_program.exists():
        return cwd_program.resolve()
    return (workspace_root / "rulespec-us" / DEFAULT_PROGRAM_RELATIVE_PATH).resolve()


def _policyengine_import_error_message(exc: BaseException) -> str:
    message = str(exc)
    if "mixes computation modes" in message:
        return (
            "policyengine-us could not be imported because policyengine-core is "
            "too new for this checkout. Run the oracle with "
            "`--with policyengine-core==3.26.0`, or refresh the PE-US checkout "
            "and lockfile together."
        )
    return (
        "policyengine-us is required. Run with: "
        "uv run --with policyengine-core==3.26.0 --with "
        f"{populace_data_requirement('us')} --with numpy axiom-encode "
        "medicaid-populace-compare"
    )


def _person_state_codes(sim: Any, period: int, count: int) -> np.ndarray:
    household_ids = calculate(sim, "household_id", period)
    household_states = calculate(sim, "state_code_str", period).astype(str)
    state_by_household_id = {
        int(household_id): str(state)
        for household_id, state in zip(household_ids, household_states, strict=False)
    }
    person_household_ids = calculate_or_default(
        sim,
        "person_household_id",
        period,
        -1,
        count,
    )
    return np.asarray(
        [
            state_by_household_id.get(int(household_id), "")
            for household_id in person_household_ids
        ]
    )


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _subset_table_by_ids(
    dataset: Any,
    *,
    table_name: str,
    id_column: str,
    ids: set[Any],
) -> pd.DataFrame:
    table = getattr(dataset, table_name)
    if table.empty or id_column not in table:
        return _empty_like(table)
    return table[table[id_column].isin(ids)].copy().reset_index(drop=True)


def _household_state_mask(household: pd.DataFrame, state_filter: str | None) -> Any:
    if state_filter is None:
        return pd.Series(True, index=household.index)
    state = state_filter.upper()
    if "state_code_str" in household:
        return household["state_code_str"].astype(str).str.upper() == state
    if "state_code" in household:
        return household["state_code"].astype(str).str.upper() == state
    if "state_fips" in household:
        state_fips = STATE_FIPS_BY_CODE.get(state)
        if state_fips is None:
            raise SystemExit(f"Unsupported state code for pre-sampling: {state}")
        return household["state_fips"].astype(int) == state_fips
    return pd.Series(True, index=household.index)


def pre_sample_households(
    dataset: Any,
    *,
    household_count: int,
    state_filter: str | None,
) -> Any:
    """Return a fresh dataset containing complete selected households."""
    require_pandas()
    if household_count <= 0:
        raise SystemExit("--pre-sample-households must be positive.")
    person = dataset.person
    household = dataset.household
    if "household_id" not in household:
        raise SystemExit("Populace household table is missing household_id.")
    if "person_household_id" not in person:
        raise SystemExit("Populace person table is missing person_household_id.")

    candidate_households = household[_household_state_mask(household, state_filter)]
    selected_household_ids = set(
        candidate_households["household_id"].head(household_count)
    )
    sampled_household = (
        household[household["household_id"].isin(selected_household_ids)]
        .copy()
        .reset_index(drop=True)
    )
    sampled_person = (
        person[person["person_household_id"].isin(selected_household_ids)]
        .copy()
        .reset_index(drop=True)
    )
    if sampled_person.empty:
        raise SystemExit(
            "Pre-sampling selected no Populace people. Check --state and "
            "--pre-sample-households."
        )

    sampled_tax_unit = _subset_table_by_ids(
        dataset,
        table_name="tax_unit",
        id_column="tax_unit_id",
        ids=set(sampled_person["person_tax_unit_id"])
        if "person_tax_unit_id" in sampled_person
        else set(),
    )
    sampled_spm_unit = _subset_table_by_ids(
        dataset,
        table_name="spm_unit",
        id_column="spm_unit_id",
        ids=set(sampled_person["person_spm_unit_id"])
        if "person_spm_unit_id" in sampled_person
        else set(),
    )
    sampled_family = _subset_table_by_ids(
        dataset,
        table_name="family",
        id_column="family_id",
        ids=set(sampled_person["person_family_id"])
        if "person_family_id" in sampled_person
        else set(),
    )
    sampled_marital_unit = _subset_table_by_ids(
        dataset,
        table_name="marital_unit",
        id_column="marital_unit_id",
        ids=set(sampled_person["person_marital_unit_id"])
        if "person_marital_unit_id" in sampled_person
        else set(),
    )
    return dataset.__class__(
        person=sampled_person,
        household=sampled_household,
        tax_unit=sampled_tax_unit,
        spm_unit=sampled_spm_unit,
        family=sampled_family,
        marital_unit=sampled_marital_unit,
        time_period=int(dataset.time_period),
    )


def _threshold_inputs_from_component(
    *,
    eligible: bool,
    low: float = 0.50,
    high: float = 2.00,
    threshold: float = 1.00,
) -> tuple[float, float]:
    return (low, threshold) if eligible else (high, threshold)


def _project_case_inputs(
    base_inputs: dict[str, Any],
    *,
    age: float,
    medicaid_income_level: float,
    parent_nfc: bool,
    parent_fc: bool,
    pregnant_nfc: bool,
    pregnant_fc: bool,
    infant_fc: bool,
    young_child_fc: bool,
    older_child_eligible: bool,
    adult_nfc: bool,
    adult_fc: bool,
    ssi_recipient: bool,
    mandatory_subpart_b: bool,
    work_requirement_eligible: bool,
    medicare_eligible: bool,
) -> dict[str, Any]:
    inputs = dict(base_inputs)
    numeric_age = float(age)
    for key in (
        "us:regulations/42-cfr/435/118#input.person_age",
        "us:regulations/42-cfr/435/119#input.person_age",
        "us:regulations/42-cfr/435/150#input.person_age",
    ):
        inputs[key] = numeric_age

    inputs[
        "us:regulations/42-cfr/435/110#input.person_is_parent_or_caretaker_relative"
    ] = bool(parent_nfc)
    inputs[
        "us:regulations/42-cfr/435/110#input.person_is_spouse_of_parent_or_caretaker_relative_living_with_them"
    ] = False
    inputs[
        "us:regulations/42-cfr/435/110#input.household_income_at_or_below_state_plan_income_standard"
    ] = bool(parent_fc)

    pregnant_full_eligible = bool(pregnant_nfc and pregnant_fc)
    pregnant_limit = _threshold_inputs_from_component(eligible=pregnant_full_eligible)[
        1
    ]
    child_component_eligible = bool(
        (numeric_age < 1 and infant_fc)
        or (1 <= numeric_age <= 5 and young_child_fc)
        or older_child_eligible
    )
    adult_full_eligible = bool(adult_nfc and adult_fc)
    # The current rules engine run request addresses inputs by name within the
    # compiled program. Several imported Medicaid modules use this same local
    # input name, so use one synthetic income projection for all of them.
    shared_income_fraction = (
        1.0
        if (pregnant_full_eligible or child_component_eligible or adult_full_eligible)
        else 2.0
    )

    _, child_limit = _threshold_inputs_from_component(
        eligible=child_component_eligible,
        threshold=1.33,
    )
    inputs["us:regulations/42-cfr/435/116#input.person_is_pregnant"] = bool(
        pregnant_nfc
    )
    inputs[
        "us:regulations/42-cfr/435/116#input.household_income_as_fraction_of_fpl"
    ] = shared_income_fraction
    inputs[
        "us:regulations/42-cfr/435/116#input.state_plan_pregnant_women_income_standard_fpl_ratio"
    ] = pregnant_limit
    inputs[
        "us:regulations/42-cfr/435/116#input.state_plan_full_medicaid_applicable_income_limit_fpl_ratio"
    ] = pregnant_limit

    inputs[
        "us:regulations/42-cfr/435/118#input.household_income_as_fraction_of_fpl"
    ] = shared_income_fraction
    inputs[
        "us:regulations/42-cfr/435/118#input.state_had_authorizing_legislation_for_higher_infant_income_standard"
    ] = bool(infant_fc)
    inputs[
        "us:regulations/42-cfr/435/118#input.state_plan_authorized_infant_income_standard_fpl_ratio"
    ] = child_limit
    inputs[
        "us:regulations/42-cfr/435/118#input.state_plan_age_group_grandfathered_income_standard_fpl_ratio"
    ] = child_limit
    inputs[
        "us:regulations/42-cfr/435/118#input.state_plan_child_income_standard_fpl_ratio"
    ] = child_limit

    inputs["us:regulations/42-cfr/435/119#input.person_is_pregnant"] = bool(
        pregnant_nfc
    )
    inputs[
        "us:regulations/42-cfr/435/119#input.person_entitled_to_or_enrolled_in_medicare_part_a_or_b"
    ] = bool(medicare_eligible)
    inputs[
        "us:regulations/42-cfr/435/119#input.person_eligible_and_enrolled_for_mandatory_medicaid_under_subpart_b"
    ] = bool(mandatory_subpart_b)
    inputs[
        "us:regulations/42-cfr/435/119#input.household_income_as_fraction_of_fpl"
    ] = shared_income_fraction
    inputs[
        "us:regulations/42-cfr/435/119#input.person_is_parent_or_caretaker_relative"
    ] = bool(parent_nfc)
    inputs[
        "us:regulations/42-cfr/435/119#input.person_lives_with_dependent_child_under_age_threshold"
    ] = False
    inputs["us:regulations/42-cfr/435/119#input.dependent_child_receiving_medicaid"] = (
        True
    )
    inputs["us:regulations/42-cfr/435/119#input.dependent_child_receiving_chip"] = False
    inputs[
        "us:regulations/42-cfr/435/119#input.dependent_child_enrolled_in_minimum_essential_coverage"
    ] = True

    inputs[
        "us:regulations/42-cfr/435/120/ssi-mandatory-group#input.person_is_aged_blind_or_disabled"
    ] = bool(ssi_recipient)
    inputs[
        "us:regulations/42-cfr/435/120/ssi-mandatory-group#input.person_is_receiving_or_deemed_receiving_ssi"
    ] = bool(ssi_recipient)

    inputs[
        "us:regulations/42-cfr/435/150#input.person_eligible_and_enrolled_for_mandatory_coverage_under_435_110_through_435_118_or_435_120_through_435_145"
    ] = bool(mandatory_subpart_b)
    inputs[
        "us:regulations/42-cfr/435/150#input.person_was_in_foster_care_under_state_or_tribe_responsibility_upon_attaining_age_18"
    ] = False
    inputs[
        "us:regulations/42-cfr/435/150#input.person_was_enrolled_in_medicaid_under_state_plan_or_1115_demonstration_upon_attaining_age_18"
    ] = False

    inputs["us:statutes/42/1396a/xx#input.monthly_work_hours"] = (
        80 if work_requirement_eligible else 0
    )
    inputs["us:statutes/42/1396a/xx#input.monthly_income"] = 0
    inputs[
        "us:statutes/42/1396a/xx#input.was_described_in_subsection_a_10_A_i_I_through_VII_for_part_or_all_of_month"
    ] = bool(mandatory_subpart_b)
    inputs[
        "us:statutes/42/1396a/xx#input.was_under_age_19_for_part_or_all_of_month"
    ] = numeric_age < 19
    inputs[
        "us:statutes/42/1396a/xx#input.was_entitled_to_or_enrolled_for_medicare_part_a_or_b_for_part_or_all_of_month"
    ] = bool(medicare_eligible)
    return inputs


def load_policyengine_cases(
    *,
    base_inputs: dict[str, Any],
    period: Period,
    state_filter: str | None,
    sample_size: int | None,
    pre_sample_households_count: int | None,
    positive_only: bool,
    populace_year: int,
) -> list[PersonCase]:
    try:
        from policyengine_us import Microsimulation
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise SystemExit(_policyengine_import_error_message(exc)) from exc

    print(f"Loading PolicyEngine Populace US {populace_year} dataset...", flush=True)
    dataset = load_populace_dataset(
        "us",
        year=populace_year,
        command="medicaid-populace-compare",
    )
    if pre_sample_households_count is not None:
        dataset = pre_sample_households(
            dataset,
            household_count=pre_sample_households_count,
            state_filter=state_filter,
        )
        print(
            "Pre-sampled "
            f"{len(dataset.household):,} households and {len(dataset.person):,} "
            "people before constructing PolicyEngine.",
            flush=True,
        )
    print("Constructing PolicyEngine simulation...", flush=True)
    sim = Microsimulation(dataset=dataset)

    year = period.year
    print(f"Calculating PolicyEngine {COMPARED_POLICYENGINE_OUTPUT}...", flush=True)
    pe_eligible = calculate(sim, COMPARED_POLICYENGINE_OUTPUT, year).astype(bool)
    count = len(pe_eligible)
    states = _person_state_codes(sim, year, count)
    mask = np.ones(count, dtype=bool)
    if state_filter:
        mask &= states == state_filter.upper()
    if positive_only:
        mask &= pe_eligible
    indices = np.flatnonzero(mask)
    if sample_size is not None:
        indices = indices[:sample_size]

    print(f"Projecting {len(indices):,} Populace people...", flush=True)
    values = {
        "age": calculate(sim, "age", year),
        "medicaid_income_level": calculate(sim, "medicaid_income_level", year),
        "medicaid_category": calculate(sim, "medicaid_category", year),
        "parent_fc": calculate(sim, "is_parent_for_medicaid_fc", year).astype(bool),
        "parent_nfc": calculate(sim, "is_parent_for_medicaid_nfc", year).astype(bool),
        "pregnant_nfc": calculate(sim, "is_pregnant_for_medicaid_nfc", year).astype(
            bool
        ),
        "pregnant_fc": calculate(sim, "is_pregnant_for_medicaid_fc", year).astype(bool),
        "infant_fc": calculate(sim, "is_infant_for_medicaid_fc", year).astype(bool),
        "young_child_fc": calculate(sim, "is_young_child_for_medicaid_fc", year).astype(
            bool
        ),
        "older_child": calculate(sim, "is_older_child_for_medicaid", year).astype(bool),
        "adult_nfc": calculate(sim, "is_adult_for_medicaid_nfc", year).astype(bool),
        "adult_fc": calculate(sim, "is_adult_for_medicaid_fc", year).astype(bool),
        "ssi": calculate(sim, "is_ssi_recipient_for_medicaid", year).astype(bool),
        "medicare": calculate_or_default(
            sim, "is_medicare_eligible", year, False, count
        ).astype(bool),
        "work": calculate_or_default(
            sim,
            "medicaid_work_requirement_eligible",
            year,
            True,
            count,
        ).astype(bool),
        "immigration": calculate_or_default(
            sim,
            "is_medicaid_immigration_status_eligible",
            year,
            True,
            count,
        ).astype(bool),
        "ca_ffyp": calculate_or_default(
            sim, "ca_ffyp_eligible", year, False, count
        ).astype(bool),
        "il_hbi": calculate_or_default(
            sim, "il_hbi_eligible", year, False, count
        ).astype(bool),
    }

    cases: list[PersonCase] = []
    for raw_index in indices:
        index = int(raw_index)
        age = float(values["age"][index])
        child_mandatory = bool(
            (age < 1 and values["infant_fc"][index])
            or (1 <= age <= 5 and values["young_child_fc"][index])
            or values["older_child"][index]
        )
        mandatory_subpart_b = bool(
            values["parent_fc"][index]
            and values["parent_nfc"][index]
            or values["pregnant_fc"][index]
            and values["pregnant_nfc"][index]
            or child_mandatory
            or values["older_child"][index]
            or values["ssi"][index]
        )
        inputs = _project_case_inputs(
            base_inputs,
            age=age,
            medicaid_income_level=float(values["medicaid_income_level"][index]),
            parent_nfc=bool(values["parent_nfc"][index]),
            parent_fc=bool(values["parent_fc"][index]),
            pregnant_nfc=bool(values["pregnant_nfc"][index]),
            pregnant_fc=bool(values["pregnant_fc"][index]),
            infant_fc=bool(values["infant_fc"][index]),
            young_child_fc=bool(values["young_child_fc"][index]),
            older_child_eligible=bool(values["older_child"][index]),
            adult_nfc=bool(values["adult_nfc"][index]),
            adult_fc=bool(values["adult_fc"][index]),
            ssi_recipient=bool(values["ssi"][index]),
            mandatory_subpart_b=mandatory_subpart_b,
            work_requirement_eligible=bool(values["work"][index]),
            medicare_eligible=bool(values["medicare"][index]),
        )
        cases.append(
            PersonCase(
                person_index=index,
                state=str(states[index]),
                inputs=inputs,
                pe_outputs={
                    COMPARED_POLICYENGINE_OUTPUT: bool(pe_eligible[index]),
                    "age": float(values["age"][index]),
                    "medicaid_income_level": float(
                        values["medicaid_income_level"][index]
                    ),
                    "medicaid_category": str(values["medicaid_category"][index]),
                    "is_parent_for_medicaid_nfc": bool(values["parent_nfc"][index]),
                    "is_parent_for_medicaid_fc": bool(values["parent_fc"][index]),
                    "is_pregnant_for_medicaid_nfc": bool(values["pregnant_nfc"][index]),
                    "is_pregnant_for_medicaid_fc": bool(values["pregnant_fc"][index]),
                    "is_infant_for_medicaid_fc": bool(values["infant_fc"][index]),
                    "is_young_child_for_medicaid_fc": bool(
                        values["young_child_fc"][index]
                    ),
                    "is_older_child_for_medicaid": bool(values["older_child"][index]),
                    "is_adult_for_medicaid_nfc": bool(values["adult_nfc"][index]),
                    "is_adult_for_medicaid_fc": bool(values["adult_fc"][index]),
                    "is_ssi_recipient_for_medicaid": bool(values["ssi"][index]),
                    "is_medicare_eligible": bool(values["medicare"][index]),
                    "medicaid_work_requirement_eligible": bool(values["work"][index]),
                    "projected_mandatory_subpart_b": mandatory_subpart_b,
                    "is_medicaid_immigration_status_eligible": bool(
                        values["immigration"][index]
                    ),
                    "ca_ffyp_eligible": bool(values["ca_ffyp"][index]),
                    "il_hbi_eligible": bool(values["il_hbi"][index]),
                },
            )
        )
    return cases


def run_axiom_person_cases(
    *,
    binary: Path,
    artifact: Path,
    cases: list[PersonCase],
    period: Period,
    env: dict[str, str],
    axiom_output_ids: list[str],
) -> list[dict[str, Any]]:
    interval = {"start": period.start.isoformat(), "end": period.end.isoformat()}
    period_json = {
        "period_kind": "month",
        "start": period.start.isoformat(),
        "end": period.end.isoformat(),
        "name": period.label,
    }
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for case in cases:
        entity_id = f"person-{case.person_index}"
        for name, value in case.inputs.items():
            inputs.append(
                {
                    "name": name,
                    "entity": "Person",
                    "entity_id": entity_id,
                    "interval": interval,
                    "value": scalar_value(value),
                }
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": period_json,
                "outputs": axiom_output_ids,
            }
        )

    request = {
        "mode": "fast",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }
    result = subprocess.run(
        [str(binary), "run-compiled", "--artifact", str(artifact)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    payload = json.loads(result.stdout)
    return payload["results"]


def runtime_output_id_for_public_id(artifact: Path, public_id: str) -> str:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    public_name = public_id.rsplit("#", 1)[-1]
    program = payload.get("program") if isinstance(payload, dict) else {}
    derived = program.get("derived") if isinstance(program, dict) else []
    matches = [
        str(output["id"])
        for output in derived
        if isinstance(output, dict)
        and output.get("id")
        and (
            output.get("name") == public_name
            or str(output.get("id")).endswith(f"#{public_name}")
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"Compiled artifact does not expose {public_id}.")
    raise RuntimeError(
        f"Compiled artifact exposes multiple outputs matching {public_id}: "
        + ", ".join(matches)
    )


def compare(
    cases: list[PersonCase],
    results: list[dict[str, Any]],
    *,
    axiom_output_id_by_label: dict[str, str],
) -> list[dict[str, Any]]:
    axiom_output_id = axiom_output_id_by_label["eligible"]
    rows: list[dict[str, Any]] = []
    for case, result in zip(cases, results, strict=True):
        raw_outputs = result.get("outputs", {})
        output_references = outputs_by_reference(raw_outputs)
        if axiom_output_id not in output_references:
            raise ValueError(
                f"Axiom result for person {case.person_index} is missing "
                f"{axiom_output_id}"
            )
        axiom_value = output_to_python(output_references[axiom_output_id])
        axiom_holds = axiom_value == "holds" or axiom_value is True
        pe_holds = bool(case.pe_outputs[COMPARED_POLICYENGINE_OUTPUT])
        row = {
            "person_index": case.person_index,
            "state": case.state,
            "pe_is_medicaid_eligible": pe_holds,
            "axiom_is_medicaid_eligible": axiom_holds,
            "match": axiom_holds == pe_holds,
            "pe_immigration_status_eligible": case.pe_outputs[
                "is_medicaid_immigration_status_eligible"
            ],
            "pe_ca_ffyp_eligible": case.pe_outputs["ca_ffyp_eligible"],
            "pe_il_hbi_eligible": case.pe_outputs["il_hbi_eligible"],
            "pe_age": case.pe_outputs["age"],
            "pe_medicaid_income_level": case.pe_outputs["medicaid_income_level"],
            "pe_medicaid_category": case.pe_outputs["medicaid_category"],
            "pe_parent_nfc": case.pe_outputs["is_parent_for_medicaid_nfc"],
            "pe_parent_fc": case.pe_outputs["is_parent_for_medicaid_fc"],
            "pe_pregnant_nfc": case.pe_outputs["is_pregnant_for_medicaid_nfc"],
            "pe_pregnant_fc": case.pe_outputs["is_pregnant_for_medicaid_fc"],
            "pe_infant_fc": case.pe_outputs["is_infant_for_medicaid_fc"],
            "pe_young_child_fc": case.pe_outputs["is_young_child_for_medicaid_fc"],
            "pe_older_child": case.pe_outputs["is_older_child_for_medicaid"],
            "pe_adult_nfc": case.pe_outputs["is_adult_for_medicaid_nfc"],
            "pe_adult_fc": case.pe_outputs["is_adult_for_medicaid_fc"],
            "pe_ssi": case.pe_outputs["is_ssi_recipient_for_medicaid"],
            "pe_medicare": case.pe_outputs["is_medicare_eligible"],
            "pe_work": case.pe_outputs["medicaid_work_requirement_eligible"],
            "projected_mandatory_subpart_b": case.pe_outputs[
                "projected_mandatory_subpart_b"
            ],
        }
        for label, output_id in axiom_output_id_by_label.items():
            if label == "eligible":
                continue
            value = output_references.get(output_id)
            rows_value = None
            if value is not None:
                python_value = output_to_python(value)
                rows_value = python_value == "holds" or python_value is True
            row[f"axiom_{label}"] = rows_value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_pe_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "category": "",
            "compared": 0,
            "pe_eligible": 0,
            "axiom_eligible": 0,
            "mismatches": 0,
            "pe_true_axiom_false": 0,
            "pe_false_axiom_true": 0,
        }
    )
    for row in rows:
        category = str(row["pe_medicaid_category"])
        item = summary[category]
        item["category"] = category
        item["compared"] += 1
        if row["pe_is_medicaid_eligible"]:
            item["pe_eligible"] += 1
        if row["axiom_is_medicaid_eligible"]:
            item["axiom_eligible"] += 1
        if not row["match"]:
            item["mismatches"] += 1
            if row["pe_is_medicaid_eligible"]:
                item["pe_true_axiom_false"] += 1
            else:
                item["pe_false_axiom_true"] += 1
    return sorted(
        summary.values(),
        key=lambda item: (
            -int(item["mismatches"]),
            -int(item["pe_eligible"]),
            str(item["category"]),
        ),
    )


def print_category_summary(rows: list[dict[str, Any]]) -> None:
    category_rows = summarize_by_pe_category(rows)
    if not category_rows:
        return
    print("PE Medicaid category summary:", flush=True)
    for item in category_rows:
        compared = int(item["compared"])
        mismatches = int(item["mismatches"])
        match_rate = (compared - mismatches) / compared if compared else 1.0
        print(
            "  "
            f"{item['category']}: compared={compared:,}; "
            f"pe_eligible={int(item['pe_eligible']):,}; "
            f"axiom_eligible={int(item['axiom_eligible']):,}; "
            f"mismatches={mismatches:,}; match_rate={match_rate:.2%}; "
            "pe_true_axiom_false="
            f"{int(item['pe_true_axiom_false']):,}; "
            "pe_false_axiom_true="
            f"{int(item['pe_false_axiom_true']):,}",
            flush=True,
        )


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    workspace_root = resolve_workspace_root(args.workspace_root)
    program = resolve_program_path(workspace_root, args.program)
    test_template = resolve_test_template_path(program, args.test_template)
    binary = resolve_axiom_binary(workspace_root, args.axiom_binary)
    if not program.exists():
        raise SystemExit(f"RuleSpec program not found: {program}")
    if not test_template.exists():
        raise SystemExit(f"RuleSpec test template not found: {test_template}")
    if not binary.exists():
        raise SystemExit(f"axiom-rules-engine binary not found: {binary}")

    period = month_period(args.year, args.month)
    base_inputs = load_base_inputs(test_template)
    cases = load_policyengine_cases(
        base_inputs=base_inputs,
        period=period,
        state_filter=args.state,
        sample_size=args.sample_size,
        pre_sample_households_count=args.pre_sample_households,
        positive_only=args.positive_only,
        populace_year=args.populace_year,
    )
    env = axiom_rules_env(program, workspace_root)
    with tempfile.TemporaryDirectory(prefix="medicaid-pe-populace-") as tmp_dir:
        artifact = Path(tmp_dir) / "program.bin"
        print(f"Compiling {program}...", flush=True)
        compile_program(binary, program, artifact, env=env)
        axiom_output_id_by_label = {
            "eligible": runtime_output_id_for_public_id(
                artifact, COMPARED_AXIOM_OUTPUT_ID
            ),
            **{
                label: runtime_output_id_for_public_id(artifact, public_id)
                for label, public_id in AXIOM_COMPONENT_OUTPUT_IDS.items()
            },
        }
        print(
            f"Resolved {COMPARED_AXIOM_OUTPUT_ID} to runtime output "
            f"{axiom_output_id_by_label['eligible']}.",
            flush=True,
        )
        print(f"Running Axiom for {len(cases):,} people...", flush=True)
        results = run_axiom_person_cases(
            binary=binary,
            artifact=artifact,
            cases=cases,
            period=period,
            env=env,
            axiom_output_ids=list(axiom_output_id_by_label.values()),
        )

    rows = compare(cases, results, axiom_output_id_by_label=axiom_output_id_by_label)
    matches = sum(1 for row in rows if row["match"])
    mismatches = len(rows) - matches
    match_rate = matches / len(rows) if rows else 1.0
    print(
        f"Compared {len(rows):,} people; matches={matches:,}; "
        f"mismatches={mismatches:,}; match_rate={match_rate:.2%}."
    )
    print_category_summary(rows)
    for row in [row for row in rows if not row["match"]][: args.max_differences]:
        print(
            "DIFF "
            f"person_index={row['person_index']} state={row['state']} "
            f"pe={row['pe_is_medicaid_eligible']} "
            f"axiom={row['axiom_is_medicaid_eligible']} "
            f"immigration_ok={row['pe_immigration_status_eligible']} "
            f"ca_ffyp={row['pe_ca_ffyp_eligible']} "
            f"il_hbi={row['pe_il_hbi_eligible']} "
            f"category={row['pe_medicaid_category']} "
            f"adult_fc={row['pe_adult_fc']} work={row['pe_work']} "
            f"axiom_child={row.get('axiom_child')} "
            f"axiom_adult={row.get('axiom_adult')} "
            f"axiom_ssi={row.get('axiom_ssi')}"
        )

    if args.write_csv is not None:
        write_csv(args.write_csv, rows)
        print(f"Wrote {args.write_csv}", flush=True)

    if args.min_match_rate is not None and match_rate < args.min_match_rate:
        raise SystemExit(
            f"Match rate {match_rate:.1%} is below required {args.min_match_rate:.1%}"
        )
    if args.fail_on_mismatch and mismatches:
        raise SystemExit(f"{mismatches} Medicaid Populace rows mismatched")


if __name__ == "__main__":
    main()

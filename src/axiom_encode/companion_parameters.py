"""What a RuleSpec companion assertion may answer, shared by both runners.

`axiom-encode test` and the validator answer an assertion on a compiled parameter
here, without running the engine, so these readings must never answer what the
engine would refuse. The engine refuses to query an indexed parameter directly,
so a table's comparison is the derived rule that reads the same cell.
"""

from datetime import date
from typing import Any

# Axiom rules engine `PeriodKind` serde variants, as accepted by the engine test
# runner the org validate-rulespec workflow invokes. Local `.test.yaml` period
# coercion must accept exactly this set — no more — so `axiom-encode validate`
# and the shared workflow agree on which fixtures are executable. Accepting a
# kind the engine rejects (e.g. `year`) makes local validate pass fixtures the
# workflow then fails as `unknown variant` (axiom-encode#1112, rulespec-dk#2).
RULESPEC_ENGINE_PERIOD_KINDS = (
    "month",
    "benefit_week",
    "tax_year",
    "custom",
)


def rulespec_engine_date(value: object) -> date:
    """Parse the one date form both Python and the engine read as the same day.

    `date.fromisoformat` alone also reads `20260101` and `2026-W01-1`, and the
    engine refuses both.
    """
    text = str(value)
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise ValueError(f"Invalid isoformat string: {text!r}")
    return parsed


def rulespec_engine_period(period: dict[str, Any]) -> dict[str, Any]:
    """Return a period mapping as written, or refuse what the engine refuses.

    The engine reads `name` as text, so an unquoted `name: 2020` refuses here. A
    parameter-only case never reaches the engine, so this check is the only one.
    """
    period_kind = period["period_kind"]
    if period_kind not in RULESPEC_ENGINE_PERIOD_KINDS:
        accepted = "|".join(RULESPEC_ENGINE_PERIOD_KINDS)
        hint = " (did you mean 'tax_year'?)" if period_kind == "year" else ""
        raise ValueError(
            f"unsupported period_kind: {period_kind!r}; the engine test "
            f"runner accepts {accepted}{hint}"
        )
    for key in ("start", "end"):
        try:
            rulespec_engine_date(period[key])
        except ValueError as error:
            raise ValueError(
                f"period {key} must be an ISO date, got {period[key]!r}"
            ) from error
    if period_kind == "custom":
        name = period.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("custom period mappings must include a text name")
    return period


def rulespec_engine_index(value: object) -> int | None:
    """Read an index as the engine's `as_index` does: a whole number or nothing."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def rulespec_index_input(inputs: dict[str, Any], reference: object) -> object | None:
    """Return the one case input an `indexed_by` reference names, or None.

    Several matches read as none, so an answer never depends on key order.
    """
    reference = str(reference)
    if reference in inputs:
        return inputs[reference]
    name = reference.split("#", 1)[-1].removeprefix("input.")
    suffixes = (f"#input.{name}", f"#{name}", f".{name}")
    matches = [
        value
        for key, value in inputs.items()
        if str(key) == name or str(key).endswith(suffixes)
    ]
    return matches[0] if len(matches) == 1 else None


def rulespec_parameter_version_at(
    parameter: dict[str, Any], period_start: object
) -> dict[str, Any]:
    """Return the version the engine reads at a period start, or refuse.

    Mirrors the engine's `lookup_parameter`: of the versions that apply at the
    start, it reads the latest `effective_from`. When none applies it refuses,
    so a version that starts later or has ended never stands in.
    """
    at = rulespec_engine_date(period_start)
    applicable = []
    for version in parameter.get("versions") or []:
        effective_to = version.get("effective_to")
        if date.fromisoformat(str(version["effective_from"])) <= at and (
            effective_to is None or at <= date.fromisoformat(str(effective_to))
        ):
            applicable.append(version)
    if not applicable:
        raise ValueError(
            f"parameter `{parameter.get('name')}` has no version at {period_start}"
        )
    # Rust's `max_by_key` keeps the last of equal keys; Python's `max` the first.
    return max(
        reversed(applicable),
        key=lambda version: date.fromisoformat(str(version["effective_from"])),
    )


def rulespec_parameter_cell(
    parameter: dict[str, Any], period_start: object, inputs: dict[str, Any]
) -> dict[str, Any]:
    """Return the scalar the engine reads for a parameter in a case, or refuse."""
    name = parameter.get("name")
    values = rulespec_parameter_version_at(parameter, period_start).get("values") or {}
    key: int | None = 0
    indexed_by = parameter.get("indexed_by")
    if indexed_by:
        raw_index = rulespec_index_input(inputs, indexed_by)
        if raw_index is None:
            raise ValueError(
                f"parameter `{name}` needs input `{indexed_by}` for indexed output"
            )
        key = rulespec_engine_index(raw_index)
        if key is None:
            raise ValueError(
                f"parameter key for `{name}` must be an integer, got {raw_index!r}"
            )
    value = values.get(str(key), values.get(key))
    if not isinstance(value, dict):
        raise ValueError(
            f"parameter `{name}` has no value for key `{key}` at {period_start}"
        )
    return value

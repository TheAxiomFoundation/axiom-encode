"""PolicyEngine population loaders for Axiom validation oracles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_US_POPULACE_YEAR = 2024
DEFAULT_UK_POPULACE_YEAR = 2023
DEFAULT_LOCAL_POPULACE_DATA_PACKAGE = (
    Path.home() / "PolicyEngine" / "populace" / "packages" / "populace-data"
)


def load_populace_dataset(
    country: str,
    *,
    year: int | None = None,
    command: str,
) -> Any:
    """Load a published Populace artifact as a PolicyEngine dataset."""
    try:
        from populace.data import load
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(populace_install_message(country, command)) from exc
    try:
        return load(country, year)
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(populace_install_message(country, command)) from exc
    except Exception as exc:  # pragma: no cover - depends on external dataset access
        year_label = "latest" if year is None else str(year)
        raise SystemExit(
            f"Failed to load Populace {country.upper()} {year_label} dataset: {exc}"
        ) from exc


def population_table(dataset: Any, name: str) -> Any:
    """Return an entity table from a PolicyEngine dataset object."""
    table = getattr(dataset, name, None)
    if table is None and hasattr(dataset, "data"):
        table = getattr(dataset.data, name, None)
    if table is None:
        raise ValueError(f"Populace dataset does not expose {name!r} table.")
    return table.copy() if hasattr(table, "copy") else table


def local_dataset_path(value: str | None) -> Path | None:
    """Return ``value`` as a local dataset path when it names an existing file."""
    if not value:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return path
    return None


def populace_install_message(country: str, command: str) -> str:
    return (
        "Populace validation requires populace-data and the matching "
        "PolicyEngine country engine. Run with: "
        f"uv run --with {populace_data_requirement(country)} "
        f"axiom-encode {command}"
    )


def populace_data_requirement(country: str) -> str:
    """Return the best uv requirement string for the local Populace data shard."""
    country = country.lower()
    override = os.environ.get("AXIOM_POPULACE_DATA_PACKAGE")
    if override:
        return f"{Path(override).expanduser()}[{country}]"
    if DEFAULT_LOCAL_POPULACE_DATA_PACKAGE.exists():
        return f"{DEFAULT_LOCAL_POPULACE_DATA_PACKAGE}[{country}]"
    return f"populace-data[{country}]"

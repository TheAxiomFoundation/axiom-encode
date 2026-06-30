"""PolicyEngine population loaders for Axiom validation oracles."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any

DEFAULT_US_POPULACE_YEAR = 2024
DEFAULT_UK_POPULACE_YEAR = 2023
DEFAULT_LOCAL_POPULACE_DATA_PACKAGE = (
    Path.home() / "PolicyEngine" / "populace" / "packages" / "populace-data"
)
DEFAULT_HUGGINGFACE_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
DEFAULT_POPULACE_YEARS = {
    "us": DEFAULT_US_POPULACE_YEAR,
    "uk": DEFAULT_UK_POPULACE_YEAR,
}
POLICYENGINE_DATASET_CLASSES = {
    "us": ("policyengine_us.data", "USSingleYearDataset"),
    "uk": ("policyengine_uk.data", "UKSingleYearDataset"),
}


def load_populace_dataset(
    country: str,
    *,
    year: int | None = None,
    command: str,
) -> Any:
    """Load a published Populace artifact as a PolicyEngine dataset."""
    country = country.lower()
    try:
        local_artifact = local_populace_artifact_path(country, year=year)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if local_artifact is not None:
        try:
            return load_policyengine_dataset_artifact(country, local_artifact)
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise SystemExit(
                policyengine_dataset_install_message(country, command)
            ) from exc
        except Exception as exc:  # pragma: no cover - depends on external dataset shape
            raise SystemExit(
                f"Failed to load local Populace {country.upper()} artifact "
                f"{local_artifact}: {exc}"
            ) from exc

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


def load_policyengine_dataset_artifact(country: str, path: Path) -> Any:
    """Load a local Populace H5 artifact using its PolicyEngine dataset class."""
    country = country.lower()
    dataset_spec = POLICYENGINE_DATASET_CLASSES.get(country)
    if dataset_spec is None:
        raise ValueError(
            f"No PolicyEngine dataset loader is configured for {country!r}."
        )
    module_name, class_name = dataset_spec
    dataset_class = getattr(import_module(module_name), class_name)
    return dataset_class(file_path=str(path))


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


def local_populace_artifact_path(
    country: str,
    *,
    year: int | None = None,
    cache_root: Path | None = None,
) -> Path | None:
    """Return the local Populace H5 artifact for ``country`` when available."""
    country = country.lower()
    resolved_year = year or DEFAULT_POPULACE_YEARS.get(country)
    for env_var in local_populace_artifact_env_vars(country, resolved_year):
        override = os.environ.get(env_var)
        if not override:
            continue
        path = Path(override).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"{env_var} points to a missing Populace artifact: {path}"
            )
        return path
    if resolved_year is None:
        return None
    return huggingface_cache_populace_artifact(
        country,
        populace_artifact_filename(country, resolved_year),
        cache_root=cache_root,
    )


def local_populace_artifact_env_vars(country: str, year: int | None) -> tuple[str, ...]:
    country_token = country.upper().replace("-", "_")
    names: list[str] = []
    if year is not None:
        names.append(f"AXIOM_POPULACE_{country_token}_{year}_H5")
    names.extend(
        [
            f"AXIOM_POPULACE_{country_token}_H5",
            "AXIOM_POPULACE_H5",
            "AXIOM_POPULACE_DATASET",
            "AXIOM_POPULACE_DATA_PATH",
        ]
    )
    return tuple(names)


def populace_artifact_filename(country: str, year: int) -> str:
    return f"populace_{country.lower()}_{int(year)}.h5"


def huggingface_cache_populace_artifact(
    country: str,
    filename: str,
    *,
    cache_root: Path | None = None,
) -> Path | None:
    """Resolve a Populace artifact already present in the Hugging Face cache."""
    root = cache_root or DEFAULT_HUGGINGFACE_CACHE_ROOT
    repo_cache = root / f"datasets--policyengine--populace-{country.lower()}"
    snapshots = repo_cache / "snapshots"
    refs = repo_cache / "refs"
    for ref in huggingface_cache_refs(refs):
        try:
            revision = ref.read_text().strip()
        except OSError:
            continue
        if not revision:
            continue
        candidate = snapshots / revision / filename
        if candidate.exists():
            return candidate
    if not snapshots.exists():
        return None
    candidates = [path for path in snapshots.glob(f"*/{filename}") if path.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def huggingface_cache_refs(refs: Path) -> tuple[Path, ...]:
    if not refs.exists():
        return ()
    main = refs / "main"
    others = sorted(
        (path for path in refs.iterdir() if path.is_file() and path != main),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if main.exists():
        return (main, *others)
    return tuple(others)


def populace_install_message(country: str, command: str) -> str:
    return (
        "Populace validation requires populace-data and the matching "
        "PolicyEngine country engine. Run with: "
        f"uv run --with {populace_data_requirement(country)} "
        f"axiom-encode {command}"
    )


def policyengine_dataset_install_message(country: str, command: str) -> str:
    return (
        "Populace validation found a local artifact but requires the matching "
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

"""Convert companion inputs using their declared or compiled scalar types."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

_QUOTED_VALUE_CONVERSIONS = {
    "bool": "bool",
    "boolean": "bool",
    "judgment": "bool",
    "integer": "number",
    "int": "number",
    "decimal": "number",
    "float": "number",
    "money": "number",
    "rate": "number",
    "number": "number",
    "text": "text",
    "string": "text",
    "date": "date",
}


def _quoted_value_conversion(dtype: Any) -> str | None:
    """Name how a dtype reads a quoted value, whatever its spelling."""
    if not isinstance(dtype, str):
        return None
    return _QUOTED_VALUE_CONVERSIONS.get(dtype.strip().lower())


def companion_scalar_value(
    value: Any,
    *,
    dtype: str | None = None,
    infer_numeric_text: bool,
) -> dict[str, Any]:
    """Convert one supplied scalar, keeping text that no type makes a number.

    Relation-role programs supply `dtype`, and a numeric one converts quoted
    numbers. Legacy programs supply none, so `infer_numeric_text` keeps each
    runner's own handling: the pipeline reads their quoted numbers as numbers,
    and the CLI keeps them as text.
    """
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, (float, Decimal)):
        return {"kind": "decimal", "value": str(value)}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, str):
        conversion = _quoted_value_conversion(dtype)
        if conversion == "text":
            return {"kind": "text", "value": value}
        stripped = value.strip()
        if conversion == "date":
            return {"kind": "date", "value": stripped}
        if conversion != "number" and not infer_numeric_text:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return {"kind": "date", "value": value}
            return {"kind": "text", "value": value}
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
            return {"kind": "date", "value": stripped}
        if re.fullmatch(r"-?\d+", stripped):
            return {"kind": "integer", "value": int(stripped)}
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", stripped):
            return {"kind": "decimal", "value": stripped}
        return {"kind": "text", "value": value}
    raise ValueError(f"unsupported scalar test value {value!r}")


def companion_input_keys(case: dict[str, Any]) -> set[str]:
    """Collect factual keys from flat, compact relation and table inputs."""
    keys = set()
    inputs = case.get("input")
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        keys.update(
                            str(name)
                            for name, child in row.items()
                            if name not in {"id", "entity_id", "tuple"}
                            and not isinstance(child, (dict, list))
                        )
            elif not isinstance(value, dict):
                keys.add(str(key))
    tables = case.get("tables")
    if isinstance(tables, dict):
        for entity, rows in tables.items():
            if not isinstance(rows, list):
                continue
            snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(entity))
            snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake)
            id_keys = {"id", "entity_id", f"{snake.replace('-', '_').lower()}_id"}
            for row in rows:
                if isinstance(row, dict):
                    keys.update(
                        str(name)
                        for name, child in row.items()
                        if name not in id_keys and not isinstance(child, (dict, list))
                    )
    return keys


def compiled_input_dtypes(payload: dict[str, Any]) -> dict[str, str]:
    """Read dtype metadata and request aliases from the compiled artifact."""
    program = payload.get("program") or {}
    interface = program.get("interface") or {}
    dtypes: dict[str, str] = {}
    for item in interface.get("inputs") or []:
        if not isinstance(item, dict):
            continue
        dtype = _quoted_value_conversion(item.get("dtype"))
        if dtype is None:
            continue
        for key in (item.get("name"), item.get("legal_id")):
            if key:
                dtypes[str(key)] = dtype
    metadata = payload.get("metadata") or {}
    for item in metadata.get("input_catalog") or []:
        if not isinstance(item, dict):
            continue
        dtype = dtypes.get(str(item.get("canonical_request_name"))) or dtypes.get(
            str(item.get("slot"))
        )
        if dtype:
            for key in item.get("request_names") or []:
                dtypes[str(key)] = dtype
    return dtypes


def compiled_input_readers(payload: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Map each slot and request name to the request names of the slot's readers."""
    metadata = payload.get("metadata") or {}
    readers: dict[str, tuple[str, ...]] = {}
    for item in metadata.get("input_catalog") or []:
        if not isinstance(item, dict):
            continue
        names = tuple(str(name) for name in item.get("request_names") or [])
        for key in (item.get("slot"), *names):
            if key:
                readers[str(key)] = names
    return readers


def _input_reference(key: str) -> str:
    """Drop a rule qualifier from an absolute input reference."""
    module, separator, fragment = key.partition("#")
    if separator and ".input." in fragment:
        return f"{module}#input.{fragment.rsplit('.input.', 1)[1]}"
    return key


def input_dtype(dtypes: dict[str, str], key: str) -> str | None:
    """Look up an input key, including rule-qualified input references."""
    return _quoted_value_conversion(dtypes.get(key)) or _quoted_value_conversion(
        dtypes.get(_input_reference(key))
    )


def declared_input_dtypes(
    module_file: Path,
    input_keys: Iterable[str],
    *,
    input_readers: Mapping[str, Sequence[str]],
    resolve_reference: Callable[[str], Path | None],
) -> dict[str, str]:
    """Resolve input dtypes from the modules that read each input.

    A module's `inputs:` block describes what its own rules read, so a
    same-named declaration in a module that never reads this input says nothing
    about it. `input_readers` holds the compiled catalog's reader names, one
    `<module>#input.<slot>` per reader. Readers that would convert a quoted
    value differently leave the input untyped, and the engine still decides
    whether the value runs.
    """
    import yaml

    if not module_file.is_file():
        return {}
    modules: dict[Path, dict[Any, str]] = {}

    def module_declarations(path: Path) -> dict[Any, str]:
        """Map each declared input to its dtype, "" for an untyped declaration."""
        if path not in modules:
            data = yaml.safe_load(path.read_text()) if path.is_file() else {}
            values = (data.get("inputs") if isinstance(data, dict) else None) or []
            if isinstance(values, dict):
                values = [
                    {**item, "name": name}
                    for name, item in values.items()
                    if isinstance(item, dict)
                ]
            modules[path] = {}
            for item in values:
                if isinstance(item, dict):
                    modules[path].setdefault(
                        item.get("name"), str(item.get("dtype") or "")
                    )
        return modules[path]

    def declaration(reference: str) -> str | None:
        """Return the reference's declared dtype, or None without a declaration."""
        if ":" in reference and "#" in reference:
            target = resolve_reference(reference)
            if target is None:
                return None
            name = reference.split("#", 1)[1].rsplit(".input.", 1)[-1]
            name = name.removeprefix("input.")
        else:
            target, name = module_file, reference
        return module_declarations(target).get(name)

    dtypes = {}
    for key in input_keys:
        key = str(key)
        dtype = declaration(key)
        if dtype is None:
            found = {
                _quoted_value_conversion(declaration(reader))
                for reader in input_readers.get(_input_reference(key), ())
            } - {None}
            dtype = found.pop() if len(found) == 1 else None
        if dtype:
            dtypes[key] = dtype
    return dtypes

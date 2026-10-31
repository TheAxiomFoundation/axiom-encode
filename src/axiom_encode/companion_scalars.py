"""Convert companion inputs using their declared or compiled scalar types."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

_INPUT_DTYPES = frozenset(
    {
        "bool",
        "boolean",
        "judgment",
        "integer",
        "int",
        "decimal",
        "float",
        "money",
        "rate",
        "number",
        "text",
        "string",
        "date",
    }
)


def _normalized_dtype(dtype: Any) -> str | None:
    if not isinstance(dtype, str):
        return None
    normalized = dtype.strip().lower()
    return normalized if normalized in _INPUT_DTYPES else None


def companion_scalar_value(
    value: Any,
    *,
    dtype: str | None = None,
    relation_role_program: bool = False,
) -> dict[str, Any]:
    """Keep typed text distinct and preserve legacy scalar inference."""
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, (float, Decimal)):
        return {"kind": "decimal", "value": str(value)}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, str):
        declared = _normalized_dtype(dtype)
        if declared in {"text", "string"}:
            return {"kind": "text", "value": value}
        stripped = value.strip()
        if declared == "date":
            return {"kind": "date", "value": stripped}
        if relation_role_program and not declared:
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
        dtype = _normalized_dtype(item.get("dtype"))
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


def input_dtype(dtypes: dict[str, str], key: str) -> str | None:
    """Look up an input key, including rule-qualified input references."""
    dtype = _normalized_dtype(dtypes.get(key))
    if dtype is not None:
        return dtype
    module, separator, fragment = key.partition("#")
    if separator and ".input." in fragment:
        return _normalized_dtype(
            dtypes.get(f"{module}#input.{fragment.rsplit('.input.', 1)[1]}")
        )
    return None


def declared_input_dtypes(
    module_file: Path,
    input_keys: Iterable[str],
    *,
    resolve_reference: Callable[[str], Path | None],
    resolve_import: Callable[[str, Path], Path | None],
) -> dict[str, str]:
    """Resolve input dtypes from local declarations and the import closure."""
    import yaml

    if not module_file.is_file():
        return {}
    payloads = {}

    def payload(path):
        if path not in payloads:
            data = yaml.safe_load(path.read_text()) if path.is_file() else {}
            payloads[path] = data if isinstance(data, dict) else {}
        return payloads[path]

    def declarations(path):
        values = payload(path).get("inputs") or []
        if isinstance(values, dict):
            values = [
                {**item, "name": name}
                for name, item in values.items()
                if isinstance(item, dict)
            ]
        return {
            str(item["name"]): str(item.get("dtype") or "")
            for item in values
            if isinstance(item, dict) and item.get("name")
        }

    dtypes = {}
    unresolved = {}
    for key in input_keys:
        key = str(key)
        if ":" in key and "#" in key:
            target = resolve_reference(key)
            if target is None:
                continue
            name = key.split("#", 1)[1].rsplit(".input.", 1)[-1]
            name = name.removeprefix("input.")
        else:
            target, name = module_file, key
        local = declarations(target)
        if name in local:
            if local[name]:
                dtypes[key] = local[name]
            continue
        unresolved.setdefault(target, {}).setdefault(name, []).append(key)
    for scope, keys_by_name in unresolved.items():
        candidates = {name: {} for name in keys_by_name}
        visited = {scope}
        pending = [scope]
        while pending:
            current = pending.pop()
            for imported in payload(current).get("imports") or []:
                if not isinstance(imported, str):
                    continue
                target = resolve_import(imported, current)
                if target is None or target in visited:
                    continue
                visited.add(target)
                pending.append(target)
                for name, dtype in declarations(target).items():
                    if name in candidates and _normalized_dtype(dtype) is not None:
                        candidates[name][target] = dtype
        for name, matches in candidates.items():
            if len(matches) > 1:
                raise ValueError(
                    f"input `{keys_by_name[name][0]}` has ambiguous imported "
                    "declarations; use its declaring module's absolute `#input` "
                    "reference"
                )
            if matches:
                dtype = next(iter(matches.values()))
                dtypes.update({key: dtype for key in keys_by_name[name]})
    return dtypes

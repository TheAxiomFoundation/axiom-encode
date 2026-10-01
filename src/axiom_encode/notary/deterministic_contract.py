"""Closed deterministic-producer extension authorized in #1662 on 2026-09-22.

Inputs, parameters and output paths are fixed by protected-base enrollment.
The public generate request selects no executable, files, or signing metadata.
"""

from __future__ import annotations

import re

from ._schema import digest, fields, nonempty, ordered_strings, relative_path

DETERMINISTIC_GENERATION = "axiom/lineage-deterministic-generation/v1"


def safe_path(value):
    return relative_path(value) and re.fullmatch(r"[A-Za-z0-9_./-]+", value) is not None


def no_ancestor_files(paths):
    names = set(paths)
    return not any(
        "/".join(path.split("/")[:index]) in names
        for path in paths
        for index in range(1, len(path.split("/")))
    )


def file_inventory(value):
    return (
        isinstance(value, list)
        and 1 <= len(value) <= 1000
        and all(
            fields(row, {"path", "sha256"})
            and safe_path(row["path"])
            and digest(row["sha256"])
            for row in value
        )
        and ordered_strings([row["path"] for row in value])
        and no_ancestor_files([row["path"] for row in value])
    )


def generator_identity(value):
    return (
        fields(value, {"name", "version", "entrypoint", "files"})
        and nonempty(value["name"])
        and nonempty(value["version"])
        and safe_path(value["entrypoint"])
        and value["entrypoint"].endswith(".py")
        and file_inventory(value["files"])
        and value["entrypoint"] in {row["path"] for row in value["files"]}
    )


def runtime_identity(value):
    return (
        fields(value, {"python_path", "tree_sha256"})
        and safe_path(value["python_path"])
        and digest(value["tree_sha256"])
    )


def parameters(value):
    # Deliberately flat strings: the reviewed adapter validates its own semantic
    # parameter vocabulary. Admission requires equality with the enrolled map.
    return (
        isinstance(value, dict)
        and len(value) <= 100
        and all(
            isinstance(key, str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", key)
            and isinstance(item, str)
            and len(item) <= 1000
            and "\0" not in item
            for key, item in value.items()
        )
    )


def output_paths(value):
    return (
        ordered_strings(value)
        and 1 <= len(value) <= 1000
        and all(safe_path(path) and path.endswith((".yaml", ".yml")) for path in value)
        and no_ancestor_files(value)
    )


def deterministic_binding(entry):
    return (
        generator_identity(entry["generator"])
        and runtime_identity(entry["runtime"])
        and file_inventory(entry["inputs"])
        and parameters(entry["parameters"])
        and output_paths(entry["outputs"])
    )

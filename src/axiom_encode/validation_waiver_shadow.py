"""Report-only comparison of the incremental waiver selection with the full audit.

The shadow period before `--scope incremental` gates a pull request runs the
full audit as today and, next to it, computes the incremental selection of the
same partition. This module compares the two and writes a record. It never
decides an exit code.

The dangerous direction is a waiver that the full audit flags and the
incremental audit would not flag: the incremental gate would then pass what the
full gate fails. Any input that this module cannot read as a complete report
gives the verdict ``inconclusive``, which counts as unsafe in the record.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SHADOW_SCHEMA = "axiom-encode/validation-waiver-incremental-shadow/v1"

VERDICT_AGREE = "agree"
VERDICT_MISSED = "incremental_missed"
VERDICT_EXTRA = "incremental_extra"
VERDICT_INCONCLUSIVE = "inconclusive"


class ShadowInputError(ValueError):
    """A report or selection that cannot be compared."""


def _report_outcomes(report: object, *, label: str) -> tuple[set[str], set[str]]:
    """Return the checked and the flagged waiver paths of one audit report.

    A report is complete only when every error it states belongs to one of its
    results. An error with no result (a ledger transition failure, a systemic
    discrepancy) says nothing about which waivers either scope flags.
    """

    if not isinstance(report, Mapping):
        raise ShadowInputError(f"{label} is not an object")
    results = report.get("results")
    if not isinstance(results, list):
        raise ShadowInputError(f"{label}.results is not a list")
    errors = report.get("errors")
    if not isinstance(errors, list) or any(
        not isinstance(error, str) for error in errors
    ):
        raise ShadowInputError(f"{label}.errors is not a list of strings")
    success = report.get("success")
    if not isinstance(success, bool):
        raise ShadowInputError(f"{label}.success is not a boolean")
    checked: set[str] = set()
    flagged: set[str] = set()
    result_errors: set[str] = set()
    for index, result in enumerate(results):
        where = f"{label}.results[{index}]"
        if not isinstance(result, Mapping):
            raise ShadowInputError(f"{where} is not an object")
        path = result.get("path")
        if not isinstance(path, str) or not path:
            raise ShadowInputError(f"{where}.path is not a non-empty string")
        if path in checked:
            raise ShadowInputError(f"{where}.path {path} occurs twice")
        result_success = result.get("success")
        if not isinstance(result_success, bool):
            raise ShadowInputError(f"{where}.success is not a boolean")
        checked.add(path)
        if not result_success:
            flagged.add(path)
        error = result.get("error")
        if error is not None:
            if not isinstance(error, str):
                raise ShadowInputError(f"{where}.error is not a string")
            result_errors.add(error)
    unattributed = [error for error in errors if error not in result_errors]
    if unattributed:
        raise ShadowInputError(
            f"{label} has {len(unattributed)} error(s) that belong to no "
            f"waiver result, first: {unattributed[0]}"
        )
    if success != (not flagged and not errors):
        raise ShadowInputError(f"{label}.success disagrees with its results")
    return checked, flagged


def _selection_paths(selection: object, *, checked: set[str]) -> tuple[str, set[str]]:
    if not isinstance(selection, Mapping):
        raise ShadowInputError("selection is not an object")
    scope = selection.get("scope")
    if scope not in {"full", "incremental"}:
        raise ShadowInputError("selection.scope must be full or incremental")
    paths = selection.get("paths")
    if not isinstance(paths, (list, tuple)) or any(
        not isinstance(path, str) or not path for path in paths
    ):
        raise ShadowInputError("selection.paths is not a list of paths")
    selected = set(paths)
    if len(selected) != len(paths):
        raise ShadowInputError("selection.paths has a duplicate path")
    outside = sorted(selected - checked)
    if outside:
        # A selection can only narrow the partition the full audit checked.
        raise ShadowInputError(
            f"selection holds {len(outside)} path(s) the full audit did not "
            f"check, first: {outside[0]}"
        )
    if scope == "full" and selected != checked:
        raise ShadowInputError("a full selection must hold every checked path")
    return scope, selected


def compare_incremental_shadow(
    *,
    full_report: object,
    selection: object | None,
    incremental_report: object | None = None,
    selection_error: str | None = None,
    selection_seconds: float | None = None,
    partition: Mapping[str, Any] | None = None,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare one partition's full audit with its incremental selection.

    Without ``incremental_report`` the incremental verdicts are derived from
    the full audit: a selected waiver is flagged exactly when the full audit
    flags it, because both runs fingerprint the same tree with the same
    verifier. With ``incremental_report`` (a separate incremental run) its own
    results are compared as well.
    """

    record: dict[str, Any] = {
        "schema": SHADOW_SCHEMA,
        "report_only": True,
        "verdict": VERDICT_INCONCLUSIVE,
        "unsafe": True,
        "missed": [],
        "extra": [],
        "errors": [],
    }
    # Every record names its partition, so a collector can require exactly
    # one record per partition. A run without partitions owns the whole ledger.
    if isinstance(full_report, Mapping) and isinstance(
        full_report.get("partition"), Mapping
    ):
        record["partition"] = dict(full_report["partition"])
    elif partition is not None:
        record["partition"] = dict(partition)
    else:
        record["partition"] = {"key": None, "count": 1}
    if binding is not None:
        record["binding"] = dict(binding)
    if selection_seconds is not None:
        record["selection_seconds"] = round(float(selection_seconds), 3)
    try:
        if not isinstance(full_report, Mapping):
            raise ShadowInputError("full report is not an object")
        if full_report.get("scope") != "full":
            raise ShadowInputError("full report scope is not full")
        checked, full_flagged = _report_outcomes(full_report, label="full report")
        record["full"] = {
            "checked": len(checked),
            "flagged": len(full_flagged),
            "success": not full_flagged,
        }
        if selection_error is not None:
            raise ShadowInputError(f"incremental selection failed: {selection_error}")
        if selection is None:
            raise ShadowInputError("incremental selection was not computed")
        scope, selected = _selection_paths(selection, checked=checked)
        incremental: dict[str, Any] = {
            "scope": scope,
            "selected": len(selected),
            "unselected": len(checked - selected),
            "derived": incremental_report is None,
        }
        fallback_reason = selection.get("fallback_reason")
        if fallback_reason is not None:
            if not isinstance(fallback_reason, str):
                raise ShadowInputError("selection.fallback_reason is not a string")
            incremental["fallback_reason"] = fallback_reason
        if incremental_report is None:
            incremental_flagged = full_flagged & selected
        else:
            incremental_checked, incremental_flagged = _report_outcomes(
                incremental_report, label="incremental report"
            )
            if incremental_checked != selected:
                raise ShadowInputError(
                    "incremental report did not check exactly the selected paths"
                )
        incremental["flagged"] = len(incremental_flagged)
        record["incremental"] = incremental
    except ShadowInputError as error:
        record["errors"].append(str(error))
        return record

    missed = sorted(full_flagged - incremental_flagged)
    extra = sorted(incremental_flagged - full_flagged)
    record["missed"] = missed
    record["missed_unselected"] = sorted(full_flagged - selected)
    record["extra"] = extra
    if missed:
        record["verdict"] = VERDICT_MISSED
    elif extra:
        record["verdict"] = VERDICT_EXTRA
    else:
        record["verdict"] = VERDICT_AGREE
    # Failing more than the full audit is a disagreement to investigate, not
    # an unsound pass; an inconclusive record stays unsafe (set above).
    record["unsafe"] = missed != []
    return record


def shadow_summary_line(record: Mapping[str, Any]) -> str:
    """One line for a log or a job summary."""

    verdict = record.get("verdict", VERDICT_INCONCLUSIVE)
    parts = [f"incremental waiver audit shadow (report only): {verdict}"]
    partition = record.get("partition")
    if isinstance(partition, Mapping) and "key" in partition:
        parts.append(f"partition {partition['key']}")
    incremental = record.get("incremental")
    full = record.get("full")
    if isinstance(incremental, Mapping) and isinstance(full, Mapping):
        parts.append(
            f"scope {incremental.get('scope')}, selected "
            f"{incremental.get('selected')} of {full.get('checked')}"
        )
        parts.append(f"full audit flagged {full.get('flagged')}")
        if "fallback_reason" in incremental:
            parts.append(f"fallback: {incremental['fallback_reason']}")
    missed = record.get("missed") or []
    extra = record.get("extra") or []
    if missed:
        parts.append(f"missed {len(missed)}, first {missed[0]}")
    if extra:
        parts.append(f"extra {len(extra)}, first {extra[0]}")
    errors = record.get("errors") or []
    if errors:
        parts.append(f"error: {errors[0]}")
    return "; ".join(parts)


def _same_file(first: Path, second: Path) -> bool:
    """Whether two existing paths are one file (device and inode)."""

    try:
        return os.path.samefile(first, second)
    except (OSError, ValueError):
        return False


def path_is_within(path: Path, container: Path) -> bool:
    """Whether ``path`` is ``container`` or lies inside it, by file identity.

    Names are not compared, so a case-insensitive filesystem or another
    spelling of the same directory cannot hide the containment. A path that
    does not exist yet is judged by its existing ancestors. The resolved path
    is walked (links followed, `link/..` taken physically, as a write would).
    """

    candidate = Path(path).absolute().resolve()
    for ancestor in (candidate, *candidate.parents):
        if _same_file(ancestor, container):
            return True
    return False


def same_output_path(first: Path, second: Path) -> bool:
    """Whether two output paths name one file.

    Two existing paths are compared by identity; otherwise the resolved
    paths are compared without case, which is conservative on a
    case-sensitive filesystem.
    """

    first, second = Path(first), Path(second)
    if first.exists() and second.exists():
        return _same_file(first, second)
    return str(first.resolve()).casefold() == str(second.resolve()).casefold()


def shadow_output_path_issue(
    path: Path, *, root: Path | None, inputs: Iterable[Path] = ()
) -> str | None:
    """Return why the shadow may not write to ``path``, or None.

    The shadow must never overwrite a file of the checked repository or of
    any input of the audit (files or directories), and never follow a
    symbolic link.
    """

    path = Path(path)
    try:
        if path.is_symlink():
            return f"{path} is a symbolic link"
        if path.exists() and not path.is_file():
            return f"{path} exists and is not a regular file"
        if root is not None and path_is_within(path, Path(root)):
            return f"{path} is inside the checked repository"
        for value in inputs:
            if path_is_within(path, Path(value)):
                return f"{path} is, or is inside, an input of the audit: {value}"
    except (OSError, ValueError) as error:
        return f"{path} cannot be checked: {error}"
    return None


def write_shadow_record(path: Path, record: Mapping[str, Any]) -> None:
    """Write the record atomically as deterministic JSON.

    The temporary file is created exclusively next to the target and removed
    when the write fails.
    """

    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def remove_stale_shadow_record(path: Path) -> bool:
    """Remove a record of an earlier run at ``path``, and nothing else.

    Only a regular file (not a link) whose content is a record of this schema
    is removed, so a failed write never leaves an older verdict behind.
    """

    path = Path(path)
    try:
        if path.is_symlink() or not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema") != SHADOW_SCHEMA:
            return False
        path.unlink()
    except (OSError, ValueError):
        return False
    return True


def append_shadow_summary(path: Path, record: Mapping[str, Any]) -> None:
    """Append one Markdown line, for example to ``$GITHUB_STEP_SUMMARY``."""

    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"- {shadow_summary_line(record)}\n")

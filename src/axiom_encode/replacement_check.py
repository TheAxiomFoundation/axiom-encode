"""Typed dry-run check for replacing a RuleSpec module that is already on main.

INACTIVE. Nothing in the encoder, a workflow or a RuleSpec repository calls
this module. It reads files and prints a report; it signs nothing, admits
nothing and changes no file. It is the first step toward a replacement lane
(axiom-encode#1881): a module that is already on main has no landing route.

A replacement swaps one existing module (``old``) for one or more new module
files (``new``): the same path with new content, or a legacy topic-named module
replaced by block-path modules. The check proves three typed properties from
the files alone, without a model and without the rules engine:

1. Interface. Every rule of ``old`` that a consumer reads is still defined by
   the new files with the same kind, entity, dtype, period, unit and
   ``indexed_by``. Consumers are modules that import ``old`` (whole module or
   ``#name``), program specs that scope ``old`` (outputs and transformations),
   and the other modules in those program scopes, which the composer binds by
   name. A rule nobody reads may be removed only when the record declares it.
2. Program bindings. A name that a scoping program spec defines and ``old``
   reads (for example ``gross_income`` bound by a transformation) is still read
   by the new files, so the binding does not silently become dead.
3. Declared behavior change. For each rule in both ``old`` and ``new``, the
   effective-dated versions are compared at every boundary date. Any
   difference must be declared in a behavior-change record with an
   ``effective_from`` on or before the first different date, and the old
   definition must hold on every earlier date (the old value is kept for
   periods before the change). A declared change with no difference is also
   an issue, so the record stays exact.

The check does not run companion tests, the oracle comparison or the
reviewer; those stay with the existing gates. Exit status: 0 when no issue is
found, 1 when issues are found, 2 for a usage or input error.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml

SCHEMA = "axiom-encode/replacement-check/v1"
RECORD_SCHEMA = "axiom-encode/replacement-behavior-change/v1"
# Rule keys that are not part of a rule's interface or definition. Every
# other key (kind, entity, dtype, period, unit, indexed_by, visibility, a
# relation's ``data_relation`` block and anything new) is interface.
NON_INTERFACE_KEYS = frozenset({"name", "source", "metadata", "summary", "versions"})
_DATE_KEYS = frozenset({"effective_from", "effective_to", "from", "to"})
_STRING_LITERAL = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MAX_BYTES = 10 * 1024 * 1024
NOT_CHECKED = (
    "companion and dependent test results (rules engine)",
    "oracle or PolicyEngine comparison",
    "reviewer model judgment and source fidelity",
    "signature, manifest and replacement receipt",
)


class ReplacementCheckError(ValueError):
    """Raised when an input cannot be read as the check requires."""


@dataclass(frozen=True)
class Issue:
    code: str
    message: str

    def payload(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _load_yaml(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReplacementCheckError(f"cannot read {label}: {path}") from exc
    if len(raw) > _MAX_BYTES:
        raise ReplacementCheckError(f"{label} exceeds the size limit: {path}")
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ReplacementCheckError(f"{label} is not valid YAML: {path}") from exc


def _module_rules(payload: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise ReplacementCheckError(f"{label} has no rules list")
    rules: dict[str, dict[str, Any]] = {}
    for rule in payload["rules"]:
        if not isinstance(rule, dict) or not isinstance(rule.get("name"), str):
            raise ReplacementCheckError(f"{label} has a rule without a name")
        if rule["name"] in rules:
            raise ReplacementCheckError(
                f"{label} defines {rule['name']} more than once"
            )
        rules[rule["name"]] = rule
    return rules


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _formula_identifiers(rules: dict[str, dict[str, Any]]) -> set[str]:
    """Identifiers in rule bodies (versions and relation blocks).

    Over-inclusive on purpose; callers intersect it with known rule names.
    """

    names: set[str] = set()
    for rule in rules.values():
        body = {
            k: v
            for k, v in rule.items()
            if k not in {"name", "source", "metadata", "summary"}
        }
        versions = body.pop("versions", None) or []
        parts: list[Any] = [body]
        for version in versions:
            if isinstance(version, dict):
                parts.append({k: v for k, v in version.items() if k not in _DATE_KEYS})
        for text in _strings(parts):
            names.update(_IDENTIFIER.findall(_STRING_LITERAL.sub(" ", text)))
    return names


def _interface(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normal_scalar(value)
        for key, value in sorted(rule.items())
        if key not in NON_INTERFACE_KEYS
    }


def _normal_text(text: str) -> str:
    """Collapse whitespace outside string literals; keep literals exactly."""

    out = []
    position = 0
    for match in _STRING_LITERAL.finditer(text):
        out.append(" ".join(text[position : match.start()].split()))
        out.append(match.group(0))
        position = match.end()
    out.append(" ".join(text[position:].split()))
    return " ".join(part for part in out if part)


def _iso(value: Any, label: str) -> str:
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, str) and _DATE.fullmatch(value.strip()):
        return value.strip()
    raise ReplacementCheckError(f"{label} is not an ISO date: {value!r}")


def _normal_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return str(Decimal(str(value)).normalize())
    if isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation:
            return _normal_text(value)
        return str(number.normalize()) if number.is_finite() else value.strip()
    if isinstance(value, dict):
        return {
            str(k): _normal_scalar(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, list):
        return [_normal_scalar(item) for item in value]
    return str(value)


def _timeline(rule: dict[str, Any], label: str) -> list[tuple[str, str | None, str]]:
    """(effective_from, effective_to or None, canonical definition) per version."""

    if "versions" not in rule:
        # A relation or other versionless rule: one definition for all dates.
        body = {k: v for k, v in rule.items() if k not in NON_INTERFACE_KEYS}
        return [("0001-01-01", None, json.dumps(_normal_scalar(body), sort_keys=True))]
    entries = []
    for index, version in enumerate(rule.get("versions") or []):
        where = f"{label} version {index}"
        if not isinstance(version, dict):
            raise ReplacementCheckError(f"{where} is not a mapping")
        start_key = "effective_from" if "effective_from" in version else "from"
        end_key = "effective_to" if "effective_to" in version else "to"
        if start_key not in version:
            raise ReplacementCheckError(f"{where} has no effective_from")
        start = _iso(version[start_key], where)
        end = (
            _iso(version[end_key], where) if version.get(end_key) is not None else None
        )
        body = {k: v for k, v in version.items() if k not in _DATE_KEYS}
        entries.append((start, end, json.dumps(_normal_scalar(body), sort_keys=True)))
    starts = [start for start, _, _ in entries]
    if len(set(starts)) != len(starts):
        raise ReplacementCheckError(f"{label} has two versions with one start date")
    return sorted(entries, key=lambda item: item[0])


def _definition_at(timeline: list[tuple[str, str | None, str]], day: str) -> str | None:
    current = None
    for start, end, body in timeline:
        if start <= day and (end is None or day <= end):
            current = body
    return current


def _next_day(day: str) -> str:
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=1)).isoformat()


def first_difference(
    old: list[tuple[str, str | None, str]], new: list[tuple[str, str | None, str]]
) -> str | None:
    """The first date on which the two timelines differ, or None.

    Both timelines are constant between boundary dates, so comparing at every
    ``effective_from`` and at the day after every ``effective_to`` is exact.
    """

    points = {"0001-01-01"}
    for start, end, _ in (*old, *new):
        points.add(start)
        if end is not None and end < "9999-12-31":
            points.add(_next_day(end))
    for day in sorted(points):
        if _definition_at(old, day) != _definition_at(new, day):
            return day
    return None


def _anchor(path: str) -> tuple[str, str]:
    """(jurisdiction directory, module path without .yaml) for a repo path."""

    parts = PurePosixPath(path).parts
    if (
        len(parts) < 3
        or path.startswith("/")
        or ".." in parts
        or not path.endswith(".yaml")
        or path.endswith(".test.yaml")
    ):
        raise ReplacementCheckError(f"not a RuleSpec module path: {path}")
    return parts[0], str(PurePosixPath(*parts[1:])).removesuffix(".yaml")


def _import_targets(
    payload: Any, jurisdiction: str
) -> list[tuple[str, str, str | None]]:
    """(jurisdiction, module, name or None) for each import entry."""

    targets = []
    if not isinstance(payload, dict) or not isinstance(payload.get("imports"), list):
        return targets
    for raw in payload["imports"]:
        if not isinstance(raw, str):
            continue
        ref, _, name = raw.strip().partition("#")
        owner, sep, module = ref.partition(":")
        if not sep:
            owner, module = jurisdiction, ref
        targets.append((owner.strip(), module.strip().strip("/"), name.strip() or None))
    return targets


def _rulespec_files(repo: Path) -> Iterable[Path]:
    for path in sorted(repo.rglob("*.yaml")):
        relative = path.relative_to(repo)
        if relative.parts[0].startswith(".") or relative.parts[0] == "programs":
            continue
        if path.name.endswith(".test.yaml") or path.is_symlink() or not path.is_file():
            continue
        yield path


def _program_scope_paths(program: dict[str, Any], spec: Path, repo: Path) -> list[str]:
    name = program.get("program")
    jurisdiction = (
        name.split("/", 1)[0]
        if isinstance(name, str)
        else spec.relative_to(repo).parts[1]
    )
    country = jurisdiction.split("-", 1)[0]
    paths = []
    scope = program.get("scope")
    if isinstance(scope, dict):
        for key, entries in scope.items():
            root = country if key == "federal" else jurisdiction
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, str):
                    paths.append(f"{root}/{entry.strip().strip('/')}.yaml")
    return paths


def _program_defined_names(program: dict[str, Any]) -> set[str]:
    names = set()
    for item in program.get("transformations") or []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])
    return names


def _program_read_names(program: dict[str, Any]) -> set[str]:
    names = set()
    for output in program.get("outputs") or []:
        if isinstance(output, str):
            names.add(output.strip())
    for item in program.get("transformations") or []:
        if isinstance(item, dict):
            body = {
                k: v
                for k, v in item.items()
                if k
                not in {
                    "name",
                    "source",
                    "pattern",
                    "entity",
                    "dtype",
                    "period",
                    "unit",
                    "effective_from",
                }
            }
            for text in _strings(body):
                names.update(_IDENTIFIER.findall(text))
    return names


def _load_record(
    path: Path | None, old_path: str
) -> tuple[dict[str, str], dict[str, str], set[str], list[Issue]]:
    """Declared changes {rule: effective_from}, removals and interface changes."""

    if path is None:
        return {}, {}, set(), []
    record = _load_yaml(path, "behavior-change record")
    findings: list[Issue] = []
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        raise ReplacementCheckError(
            f"behavior-change record schema must be {RECORD_SCHEMA}"
        )
    if record.get("replaced") != old_path:
        findings.append(
            Issue(
                "record_wrong_module",
                f"record names {record.get('replaced')!r}, not {old_path}",
            )
        )
    source = record.get("source_change")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("citation"), str)
        or not source["citation"].strip()
    ):
        findings.append(
            Issue("record_missing_source", "record needs source_change.citation")
        )
    changes: dict[str, str] = {}
    for item in record.get("changes") or []:
        if not isinstance(item, dict) or not isinstance(item.get("output"), str):
            raise ReplacementCheckError("each declared change needs an output")
        name = item["output"]
        if name in changes:
            findings.append(
                Issue("record_duplicate", f"{name} is declared more than once")
            )
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            findings.append(
                Issue("record_missing_reason", f"declared change {name} has no reason")
            )
        changes[name] = _iso(item.get("effective_from"), f"declared change {name}")
    interface_changes: set[str] = set()
    for item in record.get("interface_changes") or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ReplacementCheckError("each interface change needs a name")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            findings.append(
                Issue(
                    "record_missing_reason",
                    f"interface change {item['name']} has no reason",
                )
            )
        interface_changes.add(item["name"])
    removed: dict[str, str] = {}
    for item in record.get("removed_exports") or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ReplacementCheckError("each removed export needs a name")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            findings.append(
                Issue(
                    "record_missing_reason",
                    f"removed export {item['name']} has no reason",
                )
            )
        removed[item["name"]] = item.get("reason") or ""
    return changes, removed, interface_changes, findings


def check_replacement(
    repo: Path,
    old_path: str,
    new_files: dict[str, Path],
    record_path: Path | None = None,
) -> dict[str, Any]:
    """Check one replacement. ``new_files`` maps repo path to candidate file."""

    repo = repo.resolve()
    if not new_files:
        raise ReplacementCheckError("at least one new module is required")
    old_jurisdiction, old_module = _anchor(old_path)
    old_file = repo / old_path
    old_raw = old_file.read_bytes() if old_file.is_file() else None
    if old_raw is None:
        raise ReplacementCheckError(f"old module is not a file in the repo: {old_path}")
    old_rules = _module_rules(_load_yaml(old_file, "old module"), old_path)

    new_rules: dict[str, dict[str, Any]] = {}
    new_owner: dict[str, str] = {}
    new_report = []
    findings: list[Issue] = []
    for path, file in sorted(new_files.items()):
        _anchor(path)
        raw = file.read_bytes()
        for name, rule in _module_rules(_load_yaml(file, "new module"), path).items():
            if name in new_rules:
                findings.append(
                    Issue(
                        "duplicate_new_rule",
                        f"{name} is defined by {new_owner[name]} and {path}",
                    )
                )
                continue
            new_rules[name] = rule
            new_owner[name] = path
        new_report.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest()})
        if path != old_path and ((repo / path).exists() or (repo / path).is_symlink()):
            findings.append(
                Issue(
                    "new_path_exists",
                    f"{path} already exists; a split may only add new module paths",
                )
            )
    in_place = set(new_files) == {old_path}
    if not in_place and old_path in new_files:
        findings.append(
            Issue(
                "mixed_replacement",
                "the old path is both kept and split; give one form",
            )
        )

    # Consumers: importers, scoping program specs, and co-scoped modules.
    consumers: dict[str, set[str]] = {}
    importers: list[str] = []
    programs: list[str] = []
    program_bound: dict[str, set[str]] = {}
    old_names = set(old_rules)

    def note(name: str, reader: str) -> None:
        consumers.setdefault(name, set()).add(reader)

    for file in _rulespec_files(repo):
        relative = file.relative_to(repo).as_posix()
        if relative == old_path or relative in new_files:
            continue
        text = file.read_text(errors="replace")
        if old_module not in text:
            continue
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ReplacementCheckError(f"cannot read importer {relative}") from exc
        targets = [
            target
            for target in _import_targets(payload, relative.split("/", 1)[0])
            if target[:2] == (old_jurisdiction, old_module)
        ]
        if not targets:
            continue
        importers.append(relative)
        whole_module = any(name is None for *_, name in targets)
        if whole_module and isinstance(payload.get("rules"), list):
            rules = _module_rules(payload, relative)
            for name in (_formula_identifiers(rules) - set(rules)) & old_names:
                note(name, relative)
        for *_, name in targets:
            if name is not None:
                note(name, relative)

    programs_root = repo / "programs"
    for spec in sorted(programs_root.rglob("*.yaml")) if programs_root.is_dir() else []:
        program = _load_yaml(spec, "program spec")
        if not isinstance(program, dict):
            continue
        scope = _program_scope_paths(program, spec, repo)
        if old_path not in scope:
            continue
        relative = spec.relative_to(repo).as_posix()
        programs.append(relative)
        for name in _program_read_names(program) & old_names:
            note(name, relative)
        bound = _program_defined_names(program) & (
            _formula_identifiers(old_rules) - old_names
        )
        if bound:
            program_bound[relative] = bound
        for member in scope:
            if member == old_path or member in new_files:
                continue
            member_file = repo / member
            if not member_file.is_file():
                continue
            member_payload = _load_yaml(member_file, "scoped module")
            if not isinstance(member_payload, dict) or not isinstance(
                member_payload.get("rules"), list
            ):
                continue
            member_rules = _module_rules(member_payload, member)
            for name in (
                _formula_identifiers(member_rules) - set(member_rules)
            ) & old_names:
                note(name, f"{member} (via {relative})")

    removed_names = sorted(old_names - set(new_rules))
    added_names = sorted(set(new_rules) - old_names)
    changes, declared_removed, declared_interface, record_findings = _load_record(
        record_path, old_path
    )
    findings.extend(record_findings)

    type_changed = []
    for name in sorted(old_names & set(new_rules)):
        before, after = _interface(old_rules[name]), _interface(new_rules[name])
        if before != after:
            fields = sorted(
                key
                for key in set(before) | set(after)
                if before.get(key) != after.get(key)
            )
            type_changed.append(
                {
                    "name": name,
                    "fields": fields,
                    "readers": sorted(consumers.get(name, ())),
                }
            )
            if name in consumers:
                findings.append(
                    Issue(
                        "export_type_changed",
                        f"{name} changed {', '.join(fields)} and is read by {', '.join(sorted(consumers[name]))}",
                    )
                )
            elif name not in declared_interface:
                findings.append(
                    Issue(
                        "undeclared_interface_change",
                        f"{name} changed {', '.join(fields)} and the record does not declare it",
                    )
                )
    changed_interfaces = {item["name"] for item in type_changed}
    for name in sorted(declared_interface - changed_interfaces):
        findings.append(
            Issue(
                "declared_interface_change_absent",
                f"{name} is declared as an interface change but its interface is the same",
            )
        )
    for name in removed_names:
        if name in consumers:
            findings.append(
                Issue(
                    "export_removed",
                    f"{name} is removed and is read by {', '.join(sorted(consumers[name]))}",
                )
            )
        elif name not in declared_removed:
            findings.append(
                Issue(
                    "undeclared_removal",
                    f"{name} is removed and the record does not declare it",
                )
            )
    for name in sorted(set(declared_removed) - set(removed_names)):
        findings.append(
            Issue(
                "declared_removal_absent",
                f"{name} is declared removed but is still defined",
            )
        )

    for program, names in sorted(program_bound.items()):
        dropped = sorted(names - _formula_identifiers(new_rules))
        if dropped:
            findings.append(
                Issue(
                    "program_binding_dropped",
                    f"{program} binds {', '.join(dropped)}, which the new module no longer reads",
                )
            )

    behavior = []
    for name in sorted(old_names & set(new_rules)):
        first = first_difference(
            _timeline(old_rules[name], f"{old_path} {name}"),
            _timeline(new_rules[name], f"{new_owner[name]} {name}"),
        )
        if first is None:
            if name in changes:
                findings.append(
                    Issue(
                        "declared_change_absent",
                        f"{name} is declared changed but its versions are equivalent",
                    )
                )
            continue
        behavior.append(
            {
                "name": name,
                "first_difference": first,
                "declared_effective_from": changes.get(name),
            }
        )
        if name not in changes:
            findings.append(
                Issue(
                    "undeclared_change",
                    f"{name} differs from {first} and the record does not declare it",
                )
            )
        elif first < changes[name]:
            findings.append(
                Issue(
                    "old_value_not_kept",
                    f"{name} differs from {first}, before its declared effective_from {changes[name]}",
                )
            )
    for name in sorted(set(changes) & set(removed_names)):
        findings.append(
            Issue(
                "declared_change_removed",
                f"{name} is declared changed but the new files remove it",
            )
        )
    for name in sorted(set(changes) - old_names):
        findings.append(
            Issue("declared_change_unknown", f"{name} is not a rule of {old_path}")
        )

    mechanical = []
    if not in_place:
        for importer in importers:
            mechanical.append({"kind": "repoint_import", "path": importer})
        for program in programs:
            mechanical.append({"kind": "program_scope", "path": program})
        if importers:
            findings.append(
                Issue(
                    "importers_not_repointed",
                    f"{len(importers)} module(s) still import {old_path}: {', '.join(importers)}",
                )
            )
        if programs:
            findings.append(
                Issue(
                    "program_scope_not_updated",
                    f"{len(programs)} program spec(s) still scope {old_path}: {', '.join(programs)}",
                )
            )

    return {
        "schema": SCHEMA,
        "authorizes_signing": False,
        "passed": not findings,
        "form": "same_path" if in_place else "split_or_move",
        "old": {"path": old_path, "sha256": hashlib.sha256(old_raw).hexdigest()},
        "new": new_report,
        "exports": {
            "preserved": sorted(old_names & set(new_rules)),
            "removed": removed_names,
            "added": added_names,
            "type_changed": type_changed,
        },
        "readers": {
            name: sorted(readers) for name, readers in sorted(consumers.items())
        },
        "importers": importers,
        "programs": programs,
        "program_bound_inputs": {
            k: sorted(v) for k, v in sorted(program_bound.items())
        },
        "behavior_changes": behavior,
        "mechanical_updates": mechanical,
        "issues": [finding.payload() for finding in findings],
        "not_checked": list(NOT_CHECKED),
    }


def _new_argument(value: str, old_path: str) -> tuple[str, Path]:
    """``REPO_PATH=FILE``, or ``FILE`` alone for a same-path replacement."""

    path, sep, file = value.partition("=")
    if not sep:
        return old_path, Path(value)
    return path, Path(file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo", type=Path, required=True, help="RuleSpec checkout (the base tree)"
    )
    parser.add_argument(
        "--old", required=True, help="repo path of the module being replaced"
    )
    parser.add_argument(
        "--new",
        action="append",
        required=True,
        help="REPO_PATH=FILE for a new module (repeat for a split); FILE alone means the same path",
    )
    parser.add_argument(
        "--record", type=Path, help="behavior-change record (YAML or JSON)"
    )
    args = parser.parse_args(argv)
    try:
        new_files: dict[str, Path] = {}
        for value in args.new:
            path, file = _new_argument(value, args.old)
            if path in new_files:
                raise ReplacementCheckError(f"new path given twice: {path}")
            new_files[path] = file
        report = check_replacement(args.repo, args.old, new_files, args.record)
    except (ReplacementCheckError, OSError) as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

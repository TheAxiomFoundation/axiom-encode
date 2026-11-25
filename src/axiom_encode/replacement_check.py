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
   ``#name``), program specs that scope ``old`` (every place a spec names a
   rule: ``outputs``, ``auto_gate_outputs``, ``acknowledged_incomplete`` and
   transformation bodies), and the other modules in those program scopes,
   which the composer binds by name. Rule ``metadata`` other than the proof
   block is interface (``metadata.private`` is a visibility change). A rule nobody reads may be removed only when the record declares it.
   A rule that the new files add must not take a name that a scoping program
   or a co-scoped module already defines. The module-level ``imports``,
   ``inputs`` and every other module-level field except provenance metadata
   are compared too; a difference must be declared in the record, and an
   ``imports`` or ``inputs`` declaration must list the exact entries.
2. Program bindings. A name that a scoping program spec defines and ``old``
   reads (for example ``gross_income`` bound by a transformation) is still read
   by the new files, so the binding does not silently become dead.
3. Declared behavior change. For each rule in both ``old`` and ``new``, the
   effective-dated versions are compared at every boundary date. Any
   difference must be declared in a behavior-change record with an
   ``effective_from`` equal to the first different date, so the old
   definition holds on every earlier date (the old value is kept for
   periods before the change) and the record does not name an earlier date
   than the real change. A declared change with no difference is also an
   issue, so the record stays exact.

The check fails closed. It refuses (exit status 2) whatever it cannot parse
or resolve: a YAML mapping with a repeated key, a path that is not the exact
canonical spelling of a file in the repo tree, a program scope entry or an
import entry that it cannot resolve, a program spec with a key or a value
shape it does not know, and a rule with a non-string key. A numeric literal
is folded (``1.50`` equals ``1.5``) only when it is a plain decimal; any
other spelling (``03``, ``1_000``, ``1e3``) is compared as text.

For a split or a move, every file that still names the old path is a
required mechanical update and an issue: importers, program scopes, the
companion ``.test.yaml`` and the top-level registry files (for example
``known-validation-gaps.yaml`` and ``oracle-coverage-pending.yaml``).

The check does not run companion tests, the oracle comparison or the
reviewer; those stay with the existing gates. It reads one repository, so
readers in other jurisdiction repositories are not checked. Exit status: 0
when no issue is found, 1 when issues are found, 2 for a usage or input
error.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
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
# ``metadata`` is interface too, except the keys below.
NON_INTERFACE_KEYS = frozenset({"name", "source", "summary", "versions"})
# Rule ``metadata`` keys that are pure provenance. On rulespec-us main the
# metadata keys are ``proof`` (proof atoms, read by proof validation only),
# ``private`` (visibility) and a few scope notes; only ``proof`` is excluded.
_RULE_METADATA_PROVENANCE_KEYS = frozenset({"proof"})
# A plain decimal literal. Only this spelling is folded to a number.
_STRICT_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_DATE_KEYS = frozenset({"effective_from", "effective_to", "from", "to"})
_STRING_LITERAL = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z0-9]{2,})*")
_MAX_BYTES = 10 * 1024 * 1024
# Module-level keys that the check handles by their own rule (``rules``,
# ``imports``, ``inputs``) and ``module`` keys that are provenance metadata.
# Every other module-level field is compared as written.
_MODULE_METADATA_KEYS = frozenset(
    {"summary", "source_verification", "proof_validation"}
)
# Input keys on rulespec-us main: name, description, dtype, entity, period,
# unit. Only ``description`` is prose; every other key is compared.
_INPUT_METADATA_KEYS = frozenset({"description"})
# Program spec keys on rulespec-us main. A spec with any other key, or with
# another value shape, is refused: the check cannot tell whether it names a
# rule there.
_PROGRAM_KEYS = frozenset(
    {
        "program",
        "period",
        "scope",
        "outputs",
        "auto_gate_outputs",
        "acknowledged_incomplete",
        "transformations",
    }
)
_PROGRAM_NAME_LISTS = ("outputs", "auto_gate_outputs", "acknowledged_incomplete")
_PROGRAM_SCOPE_KEYS = frozenset({"federal", "state"})
# Transformation keys that label or type the transformed name.
_TRANSFORMATION_LABEL_KEYS = frozenset(
    {
        "name",
        "source",
        "pattern",
        "entity",
        "dtype",
        "period",
        "unit",
        "effective_from",
        "arity",
    }
)
# Transformation keys whose text names the rules that the transformation reads.
_TRANSFORMATION_READ_KEYS = frozenset(
    {
        "condition",
        "conditions",
        "formula",
        "relation",
        "terms",
        "when_true",
        "when_false",
    }
)
NOT_CHECKED = (
    "companion and dependent test results (rules engine)",
    "oracle or PolicyEngine comparison",
    "reviewer model judgment and source fidelity",
    "signature, manifest and replacement receipt",
    "readers in other jurisdiction repositories (for example rulespec-us-tn, "
    "rulespec-us-nc): only the given checkout is scanned",
    "YAML outside jurisdiction directories, program specs and top-level registry files",
    "companion tests of other modules and Python tests that name the old module",
)


class ReplacementCheckError(ValueError):
    """Raised when an input cannot be read as the check requires."""


@dataclass(frozen=True)
class Issue:
    code: str
    message: str

    def payload(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class _DuplicateKeyError(yaml.YAMLError):
    """A YAML mapping repeats a key."""


class _UniqueKeyLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    """A safe loader that refuses a mapping with a repeated key.

    ``yaml.safe_load`` keeps the last value of a repeated key, so a file
    could show one value to a reader and another to this check.
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                repeated = key in seen
            except TypeError as exc:
                raise _DuplicateKeyError("a mapping key is not hashable") from exc
            if repeated:
                raise _DuplicateKeyError(f"key {key!r} is repeated")
            seen.add(key)
        return super().construct_mapping(node, deep)

    def _construct_number(self, node: Any) -> Any:
        """Keep the spelling of a number that is not a plain decimal.

        YAML 1.1 reads ``03`` as 3, ``010`` as 8 and ``1_000`` as 1000. A
        reader with other rules can see another value, so only a plain
        decimal becomes a number here; any other spelling stays text.
        """

        text = self.construct_scalar(node)
        if not isinstance(text, str) or not _STRICT_NUMBER.fullmatch(text):
            return text
        return Decimal(text) if "." in text else int(text)


for _tag in ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float"):
    _UniqueKeyLoader.add_constructor(_tag, _UniqueKeyLoader._construct_number)


def _load_yaml(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReplacementCheckError(f"cannot read {label}: {path}") from exc
    if len(raw) > _MAX_BYTES:
        raise ReplacementCheckError(f"{label} exceeds the size limit: {path}")
    try:
        return yaml.load(raw, Loader=_UniqueKeyLoader)  # noqa: S506 - safe subclass
    except _DuplicateKeyError as exc:
        raise ReplacementCheckError(
            f"{label} repeats a YAML key ({exc}): {path}"
        ) from exc
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        raise ReplacementCheckError(f"{label} is not valid YAML: {path}") from exc


def _module_rules(payload: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise ReplacementCheckError(f"{label} has no rules list")
    rules: dict[str, dict[str, Any]] = {}
    for rule in payload["rules"]:
        if not isinstance(rule, dict) or not isinstance(rule.get("name"), str):
            raise ReplacementCheckError(f"{label} has a rule without a name")
        if not all(isinstance(key, str) for key in rule):
            raise ReplacementCheckError(
                f"{label} rule {rule['name']} has a key that is not a string"
            )
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


def _interface_metadata(metadata: Any) -> Any:
    """Rule metadata without the pure-provenance keys; None when none is left."""

    if not isinstance(metadata, dict):
        return metadata
    kept = {
        key: value
        for key, value in metadata.items()
        if key not in _RULE_METADATA_PROVENANCE_KEYS
    }
    return kept or None


def _interface(rule: dict[str, Any]) -> dict[str, Any]:
    interface = {}
    for key, value in sorted(rule.items()):
        if key in NON_INTERFACE_KEYS:
            continue
        if key == "metadata":
            value = _interface_metadata(value)
            if value is None:
                continue
        interface[key] = _normal_scalar(value)
    return interface


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


def _normal_number(text: str) -> str:
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    return format(number.normalize(), "f") if number.is_finite() else text


def _normal_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, Decimal)):
        return _normal_number(str(value))
    if isinstance(value, str):
        # ``Decimal()`` also accepts ``1.8_5``, ``03``, ``1e3`` and ``+1``.
        # Only a plain decimal is folded; any other spelling is text.
        if _STRICT_NUMBER.fullmatch(value.strip()):
            return _normal_number(value.strip())
        return _normal_text(value)
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
        # Metadata is compared as interface, not as a dated definition.
        body = {
            k: v
            for k, v in rule.items()
            if k not in NON_INTERFACE_KEYS and k != "metadata"
        }
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

    if (
        not _is_canonical(path)
        or path.count("/") < 2
        or not path.endswith(".yaml")
        or path.endswith(".test.yaml")
    ):
        raise ReplacementCheckError(
            f"not a canonical RuleSpec module path (no './', '//', '..' or "
            f"leading '/'): {path}"
        )
    parts = PurePosixPath(path).parts
    return parts[0], str(PurePosixPath(*parts[1:])).removesuffix(".yaml")


def _is_canonical(path: str) -> bool:
    """True when ``path`` is a relative POSIX path with one spelling."""

    return (
        bool(path)
        and path == path.strip()
        and "\\" not in path
        and all(part not in {"", ".", ".."} for part in path.split("/"))
    )


class _Tree:
    """Exact-case lookups in the repo tree.

    ``Path.is_file`` follows the file system, which ignores case on macOS
    and follows symlinks. A path is a file of the tree only when every
    component is listed in its directory with the same spelling and is not
    a symlink.
    """

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._names: dict[Path, frozenset[str]] = {}

    def _listed(self, directory: Path) -> frozenset[str]:
        if directory not in self._names:
            try:
                self._names[directory] = frozenset(os.listdir(directory))
            except OSError:
                self._names[directory] = frozenset()
        return self._names[directory]

    def file(self, path: str) -> Path | None:
        if not _is_canonical(path):
            return None
        current = self.repo
        for part in path.split("/"):
            if part not in self._listed(current):
                return None
            current = current / part
            if current.is_symlink():
                return None
        return current if current.is_file() else None


def _import_targets(
    payload: Any, jurisdiction: str, label: str
) -> list[tuple[str, str, str | None]]:
    """(jurisdiction, module, name or None) for each import entry."""

    targets: list[tuple[str, str, str | None]] = []
    if not isinstance(payload, dict) or payload.get("imports") is None:
        return targets
    if not isinstance(payload["imports"], list):
        raise ReplacementCheckError(f"{label} has imports that are not a list")
    for raw in payload["imports"]:
        if not isinstance(raw, str):
            raise ReplacementCheckError(
                f"{label} has an import entry that is not a string: {raw!r}"
            )
        ref, marker, name = raw.strip().partition("#")
        owner, sep, module = ref.partition(":")
        if not sep:
            owner, module = jurisdiction, ref
        owner, module, name = owner.strip(), module.strip(), name.strip()
        if (
            not _is_canonical(owner)
            or "/" in owner
            or not _is_canonical(module)
            or module.endswith(".yaml")
            or (marker and not _IDENTIFIER.fullmatch(name))
        ):
            raise ReplacementCheckError(
                f"{label} has an import entry the check cannot resolve: {raw!r}"
            )
        targets.append((owner, module, name or None))
    return targets


def _import_text(target: tuple[str, str, str | None]) -> str:
    owner, module, name = target
    return f"{owner}:{module}" + (f"#{name}" if name else "")


def _module_inputs(payload: dict[str, Any], label: str) -> dict[str, str]:
    """{input name: canonical declaration} for the module-level inputs."""

    raw = payload.get("inputs")
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ReplacementCheckError(f"{label} has inputs that are not a list")
    inputs: dict[str, str] = {}
    for item in raw:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not all(isinstance(key, str) for key in item)
        ):
            raise ReplacementCheckError(f"{label} has an input the check cannot read")
        if item["name"] in inputs:
            raise ReplacementCheckError(
                f"{label} declares input {item['name']} more than once"
            )
        body = {
            k: v
            for k, v in item.items()
            if k != "name" and k not in _INPUT_METADATA_KEYS
        }
        inputs[item["name"]] = json.dumps(_normal_scalar(body), sort_keys=True)
    return inputs


def _module_fields(payload: dict[str, Any], label: str) -> dict[str, str]:
    """Canonical module-level fields other than rules, imports and inputs."""

    fields: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ReplacementCheckError(f"{label} has a key that is not a string")
        if key in {"rules", "imports", "inputs"}:
            continue
        if key == "module" and value is not None:
            if not isinstance(value, dict) or not all(
                isinstance(sub, str) for sub in value
            ):
                raise ReplacementCheckError(
                    f"{label} has a module block it cannot read"
                )
            for sub, item in value.items():
                if sub not in _MODULE_METADATA_KEYS:
                    fields[f"module.{sub}"] = json.dumps(
                        _normal_scalar(item), sort_keys=True
                    )
        else:
            fields[key] = json.dumps(_normal_scalar(value), sort_keys=True)
    return fields


def _rulespec_files(repo: Path) -> Iterable[Path]:
    """Module files: YAML under a jurisdiction directory (``us``, ``us-az``).

    Other top-level directories (``programs``, ``bulk``, ``tests``, ``tools``)
    and top-level files hold no modules; an import resolves to
    ``<jurisdiction>/<module>.yaml`` only.
    """

    for path in sorted(repo.rglob("*.yaml")):
        relative = path.relative_to(repo)
        if len(relative.parts) < 2 or not _JURISDICTION.fullmatch(relative.parts[0]):
            continue
        if path.name.endswith(".test.yaml") or path.is_symlink() or not path.is_file():
            continue
        yield path


def _check_program_shape(program: dict[str, Any], label: str) -> None:
    """Refuse a program spec with a key or a value shape the check does not know."""

    unknown = sorted(repr(key) for key in program if key not in _PROGRAM_KEYS)
    if unknown:
        raise ReplacementCheckError(
            f"{label} has program keys the check does not know: {', '.join(unknown)}"
        )
    for key in _PROGRAM_NAME_LISTS:
        entries = program.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list) or not all(
            isinstance(entry, str) and _IDENTIFIER.fullmatch(entry) for entry in entries
        ):
            raise ReplacementCheckError(
                f"{label} has {key} that is not a list of rule names"
            )
    transformations = program.get("transformations")
    if transformations is None:
        return
    if not isinstance(transformations, list):
        raise ReplacementCheckError(f"{label} has transformations that are not a list")
    for item in transformations:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ReplacementCheckError(f"{label} has a transformation without a name")
        unknown = sorted(
            repr(key)
            for key in item
            if key not in _TRANSFORMATION_LABEL_KEYS | _TRANSFORMATION_READ_KEYS
        )
        if unknown:
            raise ReplacementCheckError(
                f"{label} transformation {item['name']} has keys the check does "
                f"not know: {', '.join(unknown)}"
            )


def _program_scope_paths(program: dict[str, Any], spec: Path, repo: Path) -> list[str]:
    name = program.get("program")
    jurisdiction = (
        name.split("/", 1)[0]
        if isinstance(name, str)
        else spec.relative_to(repo).parts[1]
    )
    country = jurisdiction.split("-", 1)[0]
    label = spec.relative_to(repo).as_posix()
    paths: list[str] = []
    scope = program.get("scope")
    if scope is None:
        return paths
    if not isinstance(scope, dict):
        raise ReplacementCheckError(f"{label} has a scope that is not a mapping")
    for key, entries in scope.items():
        if key not in _PROGRAM_SCOPE_KEYS:
            raise ReplacementCheckError(
                f"{label} has a scope key the check does not know: {key!r}"
            )
        root = country if key == "federal" else jurisdiction
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ReplacementCheckError(f"{label} scope {key!r} is not a list")
        for entry in entries:
            # An entry names a module without the suffix. An entry with a
            # suffix, or any spelling that is not canonical, would be missed
            # by a string compare, so the check refuses it.
            if (
                not isinstance(entry, str)
                or not _is_canonical(entry)
                or entry.endswith(".yaml")
            ):
                raise ReplacementCheckError(
                    f"{label} has a scope entry the check cannot resolve: {entry!r}"
                )
            paths.append(f"{root}/{entry}.yaml")
    return paths


def _program_defined_names(program: dict[str, Any]) -> set[str]:
    names = set()
    for item in program.get("transformations") or []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])
    return names


def _program_read_names(program: dict[str, Any]) -> set[str]:
    """Every rule name a program spec names. ``_check_program_shape`` ran first."""

    names = set()
    for key in _PROGRAM_NAME_LISTS:
        names.update(program.get(key) or [])
    for item in program.get("transformations") or []:
        body = {k: v for k, v in item.items() if k in _TRANSFORMATION_READ_KEYS}
        for text in _strings(body):
            names.update(_IDENTIFIER.findall(text))
    return names


def _all_strings(value: Any) -> Iterable[str]:
    """Every string in a YAML value, mapping keys included."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(key)
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _registry_mentions(repo: Path, old_path: str, legal_id: str) -> dict[str, int]:
    """{top-level YAML file: number of strings that name the old module}.

    The top-level files are the registries (``known-validation-gaps.yaml``
    keys a module by repo path, ``oracle-coverage-pending.yaml`` by
    ``jurisdiction:module#name``). Each one is parsed; one the check cannot
    parse is refused.
    """

    spellings = {old_path, old_path.removesuffix(".yaml"), legal_id}
    mentions: dict[str, int] = {}
    for name in sorted(os.listdir(repo)):
        if not name.endswith((".yaml", ".yml")):
            continue
        path = repo / name
        if path.is_symlink() or not path.is_file():
            raise ReplacementCheckError(
                f"top-level {name} is not a regular file; the check cannot read it"
            )
        count = sum(
            1
            for text in _all_strings(_load_yaml(path, "registry file"))
            if text.strip() in spellings or text.strip().startswith(legal_id + "#")
        )
        if count:
            mentions[name] = count
    return mentions


def _load_record(
    path: Path | None, old_path: str
) -> tuple[
    dict[str, str],
    dict[str, str],
    set[str],
    dict[str, dict[str, list[str]]],
    list[Issue],
]:
    """Declared changes {rule: effective_from}, removals, interface changes
    and module-level field changes."""

    if path is None:
        return {}, {}, set(), {}, []
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
    module_changes: dict[str, dict[str, list[str]]] = {}
    for item in record.get("module_changes") or []:
        if not isinstance(item, dict) or not isinstance(item.get("field"), str):
            raise ReplacementCheckError("each module change needs a field")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            findings.append(
                Issue(
                    "record_missing_reason",
                    f"module change {item['field']} has no reason",
                )
            )
        field = item["field"]
        # An ``imports`` or ``inputs`` declaration lists the exact entries.
        entry_keys = {
            "imports": ("added", "removed"),
            "inputs": ("added", "removed", "changed"),
        }.get(field, ())
        unread = sorted(
            repr(key) for key in item if key not in {"field", "reason", *entry_keys}
        )
        if unread:
            raise ReplacementCheckError(
                f"module change {field} has keys the check does not read: "
                f"{', '.join(unread)}"
            )
        detail: dict[str, list[str]] = {}
        for key in entry_keys:
            entries = item.get(key)
            entries = [] if entries is None else entries
            if (
                not isinstance(entries, list)
                or not all(isinstance(entry, str) for entry in entries)
                or len(set(entries)) != len(entries)
            ):
                raise ReplacementCheckError(
                    f"module change {field} needs {key} as a list of distinct strings"
                )
            detail[key] = sorted(entries)
        if field in module_changes:
            findings.append(
                Issue("record_duplicate", f"{field} is declared more than once")
            )
        module_changes[field] = detail
    return changes, removed, interface_changes, module_changes, findings


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
    tree = _Tree(repo)
    old_file = tree.file(old_path)
    if old_file is None:
        raise ReplacementCheckError(
            f"old module is not a file in the repo under this exact path: {old_path}"
        )
    old_raw = old_file.read_bytes()
    old_payload = _load_yaml(old_file, "old module")
    old_rules = _module_rules(old_payload, old_path)
    old_imports = set(_import_targets(old_payload, old_jurisdiction, old_path))
    old_inputs = _module_inputs(old_payload, old_path)
    old_fields = _module_fields(old_payload, old_path)

    new_rules: dict[str, dict[str, Any]] = {}
    new_owner: dict[str, str] = {}
    new_report = []
    findings: list[Issue] = []
    new_anchors = {_anchor(path) for path in new_files}
    new_imports: set[tuple[str, str, str | None]] = set()
    new_inputs: dict[str, set[str]] = {}
    field_changes: dict[str, str] = {}
    for path, file in sorted(new_files.items()):
        raw = file.read_bytes()
        new_payload = _load_yaml(file, "new module")
        path_rules = _module_rules(new_payload, path)
        # An import of another new file of the same split is not an import
        # of the replaced module's surroundings.
        new_imports.update(
            target
            for target in _import_targets(new_payload, path.split("/", 1)[0], path)
            if target[:2] not in new_anchors
        )
        for name, declaration in _module_inputs(new_payload, path).items():
            new_inputs.setdefault(name, set()).add(declaration)
        path_fields = _module_fields(new_payload, path)
        for key in sorted(set(old_fields) | set(path_fields)):
            if old_fields.get(key) != path_fields.get(key):
                field_changes.setdefault(key, f"{key} differs in {path}")
        for name, rule in path_rules.items():
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
    # Names that a scoping program or a co-scoped module defines. The
    # composer binds by name, so a new rule must not take one of them.
    taken: dict[str, set[str]] = {}

    def note(name: str, reader: str) -> None:
        consumers.setdefault(name, set()).add(reader)

    for file in _rulespec_files(repo):
        relative = file.relative_to(repo).as_posix()
        if relative == old_path or relative in new_files:
            continue
        # Every module is parsed. A text search for the module path would
        # miss an import written with another spelling or a YAML escape.
        payload = _load_yaml(file, "module")
        targets = [
            target
            for target in _import_targets(payload, relative.split("/", 1)[0], relative)
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
        relative = spec.relative_to(repo).as_posix()
        if not isinstance(program, dict):
            raise ReplacementCheckError(f"program spec is not a mapping: {relative}")
        _check_program_shape(program, relative)
        scope = _program_scope_paths(program, spec, repo)
        for member in scope:
            if tree.file(member) is None:
                raise ReplacementCheckError(
                    f"{relative} scopes {member}, which is not a file in the repo; "
                    "the check cannot tell whether it scopes the old module"
                )
        if old_path not in scope:
            continue
        programs.append(relative)
        for name in sorted(_program_defined_names(program)):
            taken.setdefault(name, set()).add(relative)
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
            member_rules = _module_rules(
                _load_yaml(repo / member, "scoped module"), member
            )
            for name in member_rules:
                taken.setdefault(name, set()).add(f"{member} (via {relative})")
            for name in (
                _formula_identifiers(member_rules) - set(member_rules)
            ) & old_names:
                note(name, f"{member} (via {relative})")

    removed_names = sorted(old_names - set(new_rules))
    added_names = sorted(set(new_rules) - old_names)
    (
        changes,
        declared_removed,
        declared_interface,
        declared_module,
        record_findings,
    ) = _load_record(record_path, old_path)
    findings.extend(record_findings)

    for name in added_names:
        if name in taken:
            findings.append(
                Issue(
                    "added_name_collides",
                    f"{name} is added but is already defined by {', '.join(sorted(taken[name]))}",
                )
            )

    # The exact entries that changed, for the fields declared per entry.
    observed: dict[str, dict[str, list[str]]] = {}
    if new_imports != old_imports:
        gone = sorted(_import_text(t) for t in old_imports - new_imports)
        come = sorted(_import_text(t) for t in new_imports - old_imports)
        observed["imports"] = {"added": come, "removed": gone}
        field_changes["imports"] = (
            f"imports removed [{', '.join(gone)}], added [{', '.join(come)}]"
        )
    input_changes = {
        "added": sorted(set(new_inputs) - set(old_inputs)),
        "removed": sorted(set(old_inputs) - set(new_inputs)),
        "changed": sorted(
            name
            for name in set(old_inputs) & set(new_inputs)
            if new_inputs[name] != {old_inputs[name]}
        ),
    }
    if any(input_changes.values()):
        observed["inputs"] = input_changes
        field_changes["inputs"] = "inputs " + ", ".join(
            f"{key} [{', '.join(names)}]" for key, names in input_changes.items()
        )
    for key, detail in sorted(field_changes.items()):
        if key not in declared_module:
            findings.append(
                Issue(
                    "undeclared_module_change",
                    f"{detail} and the record does not declare it",
                )
            )
        elif declared_module[key] != observed.get(key, {}):
            declared = ", ".join(
                f"{name} [{', '.join(entries)}]"
                for name, entries in declared_module[key].items()
            )
            findings.append(
                Issue(
                    "module_change_declaration_inexact",
                    f"{detail}, but the record declares {declared}; the declared "
                    "entries must equal the observed entries",
                )
            )
    for key in sorted(set(declared_module) - set(field_changes)):
        findings.append(
            Issue(
                "declared_module_change_absent",
                f"{key} is declared as a module change but it is the same",
            )
        )

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
        elif first > changes[name]:
            findings.append(
                Issue(
                    "declared_effective_from_too_early",
                    f"{name} first differs on {first}, after its declared effective_from {changes[name]}",
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
        companion = old_path.removesuffix(".yaml") + ".test.yaml"
        if os.path.lexists(repo / companion):
            mechanical.append({"kind": "companion_test", "path": companion})
            findings.append(
                Issue(
                    "companion_test_not_moved",
                    f"{companion} still tests {old_path}",
                )
            )
        legal_id = f"{old_jurisdiction}:{old_module}"
        for name, count in _registry_mentions(repo, old_path, legal_id).items():
            mechanical.append({"kind": "registry_entry", "path": name, "count": count})
            findings.append(
                Issue(
                    "registry_not_updated",
                    f"{name} still names {old_path} in {count} place(s)",
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
        "module_changes": [
            {"field": key, "detail": detail}
            for key, detail in sorted(field_changes.items())
        ],
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
    except Exception as exc:  # noqa: BLE001 - refuse with JSON, never a traceback
        error = f"refused: the check failed on an input it cannot handle ({type(exc).__name__})"
        print(json.dumps({"schema": SCHEMA, "error": error}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

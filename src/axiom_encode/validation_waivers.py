"""Strict, expiring expected-failure waivers for RuleSpec validation.

The waiver file is deliberately a small state machine rather than an
allowlist.  An ``active`` record tolerates one exact combined validation and
companion-test failure.  A ``pending`` record preapproves the exact record that
may replace it in a later pull request; pending records never waive failures.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.events import (
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    NodeEvent,
    ScalarEvent,
)

MAX_WAIVER_DAYS = 90
MAX_WAIVER_FILE_BYTES = 2_000_000
MAX_WAIVER_ENTRIES = 10_000
MAX_YAML_EVENTS = 250_000
MAX_YAML_DEPTH = 32
MAX_YAML_SCALAR_LENGTH = 65_536

FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
OWNER_RE = re.compile(r"^@[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
ISSUE_RE = re.compile(
    r"^https://github\.com/TheAxiomFoundation/"
    r"[A-Za-z0-9_.-]+/issues/[1-9][0-9]*$"
)
EXPIRY_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
JURISDICTION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)*$")
CONTENT_ROOTS = frozenset({"statutes", "regulations", "policies", "legislation"})
ENTRY_KEYS = frozenset({"active", "pending"})
METADATA_KEYS = frozenset({"fingerprint", "owner", "issue", "expires"})
WAIVER_SECTION = "validate_failures"
DEFAULT_WAIVER_PATH = "known-validation-gaps.yaml"
OUTCOME_SCHEMA = "rulespec-validation-failure/v1"


class ValidationWaiverError(ValueError):
    """Base error for invalid waiver files, transitions, and outcomes."""


class WaiverSchemaError(ValidationWaiverError):
    """Raised when the strict waiver YAML is malformed."""


class WaiverTransitionError(ValidationWaiverError):
    """Raised when a head waiver set bypasses protected-base approval."""


@dataclass(frozen=True, slots=True)
class WaiverMetadata:
    fingerprint: str
    owner: str
    issue: str
    expires: str

    @property
    def expiry_date(self) -> date:
        return date.fromisoformat(self.expires)


@dataclass(frozen=True, slots=True)
class WaiverEntry:
    path: str
    active: WaiverMetadata | None = None
    pending: WaiverMetadata | None = None


@dataclass(frozen=True, slots=True)
class ValidationWaiverSet:
    entries: Mapping[str, WaiverEntry]

    @property
    def active_paths(self) -> frozenset[str]:
        return frozenset(path for path, entry in self.entries.items() if entry.active)

    @property
    def pending_paths(self) -> frozenset[str]:
        return frozenset(path for path, entry in self.entries.items() if entry.pending)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that refuses duplicate and merged mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "<<":
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are not allowed",
                key_node.start_mark,
            )
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _scan_bounded_yaml(text: str, *, source: Path) -> None:
    depth = 0
    events = 0
    try:
        stream = yaml.parse(text, Loader=yaml.SafeLoader)
        for event in stream:
            events += 1
            if events > MAX_YAML_EVENTS:
                raise WaiverSchemaError(
                    f"{source}: YAML exceeds {MAX_YAML_EVENTS} parser events"
                )
            if isinstance(event, AliasEvent):
                raise WaiverSchemaError(f"{source}: YAML aliases are not allowed")
            if isinstance(event, NodeEvent) and event.anchor is not None:
                raise WaiverSchemaError(f"{source}: YAML anchors are not allowed")
            if (
                isinstance(event, ScalarEvent)
                and len(event.value) > MAX_YAML_SCALAR_LENGTH
            ):
                raise WaiverSchemaError(
                    f"{source}: YAML scalar exceeds {MAX_YAML_SCALAR_LENGTH} characters"
                )
            if isinstance(event, CollectionStartEvent):
                depth += 1
                if depth > MAX_YAML_DEPTH:
                    raise WaiverSchemaError(
                        f"{source}: YAML nesting exceeds {MAX_YAML_DEPTH} levels"
                    )
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
    except yaml.YAMLError as error:
        raise WaiverSchemaError(f"cannot parse {source}: {error}") from error


def _load_strict_yaml(path: Path) -> object:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise WaiverSchemaError(f"{path}: waiver file must be a regular file")
        size = metadata.st_size
        if size > MAX_WAIVER_FILE_BYTES:
            raise WaiverSchemaError(
                f"{path}: file exceeds {MAX_WAIVER_FILE_BYTES} bytes"
            )
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise WaiverSchemaError(f"cannot read {path}: {error}") from error
    _scan_bounded_yaml(text, source=path)
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise WaiverSchemaError(f"cannot parse {path}: {error}") from error


def _contains_control_or_format(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    )


def _module_path(raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise WaiverSchemaError(f"{WAIVER_SECTION} keys must be non-empty strings")
    if (
        raw_path != raw_path.strip()
        or "\\" in raw_path
        or _contains_control_or_format(raw_path)
    ):
        raise WaiverSchemaError(f"unsafe {WAIVER_SECTION} path: {raw_path!r}")
    pure = PurePosixPath(raw_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != raw_path
        or any(part in {"", ".", ".."} for part in pure.parts)
        or len(pure.parts) < 3
        or JURISDICTION_RE.fullmatch(pure.parts[0]) is None
        or pure.parts[1] not in CONTENT_ROOTS
        or pure.suffix != ".yaml"
        or pure.name.endswith(".test.yaml")
    ):
        raise WaiverSchemaError(f"unsafe {WAIVER_SECTION} path: {raw_path!r}")
    return raw_path


def _require_regular_module(repo_root: Path, module_path: str) -> None:
    parts = PurePosixPath(module_path).parts
    current = repo_root
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise WaiverSchemaError(
                f"{WAIVER_SECTION} path does not exist: {module_path}"
            ) from error
        if current.is_symlink():
            raise WaiverSchemaError(
                f"{WAIVER_SECTION} path contains a symlink alias: {module_path}"
            )
        expected_type = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
        if not expected_type(metadata.st_mode):
            raise WaiverSchemaError(
                f"{WAIVER_SECTION} path is not a regular module path: {module_path}"
            )
    module = current
    root = repo_root.resolve()
    resolved = module.resolve()
    if root not in resolved.parents:
        raise WaiverSchemaError(
            f"{WAIVER_SECTION} path escapes the repo: {module_path}"
        )


def resolve_validation_waiver_module(
    repo_root: Path,
    raw_path: str,
) -> tuple[Path, str]:
    """Resolve a module using exactly the path rules accepted by the schema."""
    root = Path(repo_root).resolve()
    module_path = _module_path(raw_path)
    _require_regular_module(root, module_path)
    return root.joinpath(*PurePosixPath(module_path).parts).resolve(), module_path


def _metadata(
    raw: object,
    *,
    path: str,
    state: str,
    today: date,
    allow_expired: bool,
) -> WaiverMetadata:
    label = f"{WAIVER_SECTION}[{path!r}].{state}"
    if not isinstance(raw, Mapping):
        raise WaiverSchemaError(f"{label} must be a mapping")
    keys = frozenset(raw)
    if keys != METADATA_KEYS:
        raise WaiverSchemaError(
            f"{label} must contain exactly {sorted(METADATA_KEYS)} "
            f"(missing={sorted(METADATA_KEYS - keys)}, "
            f"extra={sorted(keys - METADATA_KEYS)})"
        )

    fingerprint = raw["fingerprint"]
    owner = raw["owner"]
    issue = raw["issue"]
    expires = raw["expires"]
    if (
        not isinstance(fingerprint, str)
        or FINGERPRINT_RE.fullmatch(fingerprint) is None
    ):
        raise WaiverSchemaError(
            f"{label}.fingerprint must be sha256:<64 lowercase hex>"
        )
    if not isinstance(owner, str) or OWNER_RE.fullmatch(owner) is None:
        raise WaiverSchemaError(f"{label}.owner must be an @GitHub-login")
    if not isinstance(issue, str) or ISSUE_RE.fullmatch(issue) is None:
        raise WaiverSchemaError(
            f"{label}.issue must be an Axiom Foundation GitHub issue URL"
        )
    if not isinstance(expires, str) or EXPIRY_RE.fullmatch(expires) is None:
        raise WaiverSchemaError(f"{label}.expires must be a quoted YYYY-MM-DD string")
    try:
        expiry = date.fromisoformat(expires)
    except ValueError as error:
        raise WaiverSchemaError(f"{label}.expires is not a valid date") from error
    if expiry > today + timedelta(days=MAX_WAIVER_DAYS):
        raise WaiverSchemaError(
            f"{label}.expires must be within {MAX_WAIVER_DAYS} days"
        )
    if not allow_expired and expiry <= today:
        raise WaiverSchemaError(f"{label} expired on {expires}")
    return WaiverMetadata(
        fingerprint=fingerprint,
        owner=owner,
        issue=issue,
        expires=expires,
    )


def load_validation_waivers(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    today: date | None = None,
    allow_expired: bool = False,
    require_paths: bool = True,
) -> ValidationWaiverSet:
    """Load the mandatory, canonical validation-waiver schema."""
    source = Path(path)
    if not source.exists():
        raise WaiverSchemaError(f"required validation waiver file is missing: {source}")
    payload = _load_strict_yaml(source)
    if not isinstance(payload, Mapping):
        raise WaiverSchemaError(f"{source}: document root must be a mapping")
    root_keys = frozenset(payload)
    if root_keys != frozenset({WAIVER_SECTION}):
        extras = sorted(repr(key) for key in root_keys if key != WAIVER_SECTION)
        raise WaiverSchemaError(
            f"{source}: document root must contain exactly {WAIVER_SECTION} "
            f"(extra={extras})"
        )
    raw_entries = payload[WAIVER_SECTION]
    if not isinstance(raw_entries, Mapping):
        raise WaiverSchemaError(
            f"{source}: {WAIVER_SECTION} must be a mapping keyed by module path"
        )
    if len(raw_entries) > MAX_WAIVER_ENTRIES:
        raise WaiverSchemaError(
            f"{source}: {WAIVER_SECTION} exceeds {MAX_WAIVER_ENTRIES} entries"
        )

    current_date = today or date.today()
    root = Path(repo_root).resolve() if repo_root is not None else None
    entries: dict[str, WaiverEntry] = {}
    for raw_path, raw_entry in raw_entries.items():
        module_path = _module_path(raw_path)
        if root is not None and require_paths:
            _require_regular_module(root, module_path)
        label = f"{WAIVER_SECTION}[{module_path!r}]"
        if not isinstance(raw_entry, Mapping):
            raise WaiverSchemaError(f"{label} must be a mapping")
        state_keys = frozenset(raw_entry)
        if not state_keys or not state_keys <= ENTRY_KEYS:
            raise WaiverSchemaError(
                f"{label} must contain only active and/or pending states"
            )
        active = (
            _metadata(
                raw_entry["active"],
                path=module_path,
                state="active",
                today=current_date,
                allow_expired=allow_expired,
            )
            if "active" in raw_entry
            else None
        )
        pending = (
            _metadata(
                raw_entry["pending"],
                path=module_path,
                state="pending",
                today=current_date,
                allow_expired=allow_expired,
            )
            if "pending" in raw_entry
            else None
        )
        entries[module_path] = WaiverEntry(
            path=module_path,
            active=active,
            pending=pending,
        )
    return ValidationWaiverSet(MappingProxyType(dict(sorted(entries.items()))))


def protected_base_transition_issues(
    base: ValidationWaiverSet,
    head: ValidationWaiverSet,
    *,
    changed_paths: set[str] | frozenset[str],
    waiver_path: str = DEFAULT_WAIVER_PATH,
    today: date | None = None,
) -> tuple[str, ...]:
    """Return attempts to activate a waiver without protected-base approval."""
    issues: list[str] = []
    current_date = today or date.today()
    changed = frozenset(changed_paths)
    for path in sorted(set(base.entries) | set(head.entries)):
        base_entry = base.entries.get(path)
        head_entry = head.entries.get(path)
        base_active = base_entry.active if base_entry else None
        base_pending = base_entry.pending if base_entry else None
        head_active = head_entry.active if head_entry else None
        head_pending = head_entry.pending if head_entry else None

        if head_active is not None and head_active != base_active:
            if base_pending is None or head_active != base_pending:
                issues.append(
                    f"{path}: new or changed active waiver must exactly consume "
                    "the protected base's pending record"
                )
            else:
                if base_pending.expiry_date <= current_date:
                    issues.append(
                        f"{path}: protected-base pending approval expired on "
                        f"{base_pending.expires}"
                    )
                if head_pending is not None:
                    issues.append(
                        f"{path}: activating a pending waiver must consume it"
                    )

        pending_added_or_changed = (
            head_pending is not None and head_pending != base_pending
        )
        if pending_added_or_changed and changed != frozenset({waiver_path}):
            issues.append(
                f"{path}: new or changed pending approval requires a waiver-only "
                f"pull request (only {waiver_path} may change)"
            )
    return tuple(issues)


def validate_protected_base_transition(
    base: ValidationWaiverSet,
    head: ValidationWaiverSet,
    *,
    changed_paths: set[str] | frozenset[str],
    waiver_path: str = DEFAULT_WAIVER_PATH,
    today: date | None = None,
) -> None:
    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths=changed_paths,
        waiver_path=waiver_path,
        today=today,
    )
    if issues:
        raise WaiverTransitionError("; ".join(issues))


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RULESPEC_ALIAS_TEMP_RE = re.compile(
    r"<system-tmp>[/\\]axiom-rulespec-repo-aliases[/\\][0-9a-f]{16}"
)
_RANDOM_TEMP_RE = re.compile(r"<system-tmp>[/\\](?:tmp|\.tmp)[A-Za-z0-9._-]+")


def _normalize_text(value: object, replacements: Mapping[str, str]) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _ANSI_RE.sub("", text)
    for raw, replacement in sorted(
        ((str(raw), str(replacement)) for raw, replacement in replacements.items()),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        if not raw:
            continue
        text = re.sub(
            rf"(?<![A-Za-z0-9_.-]){re.escape(raw)}(?=$|[^A-Za-z0-9_.-])",
            lambda _match: replacement,
            text,
        )
        slash_raw = raw.replace("\\", "/")
        if slash_raw != raw:
            text = re.sub(
                rf"(?<![A-Za-z0-9_.-]){re.escape(slash_raw)}"
                r"(?=$|[^A-Za-z0-9_.-])",
                lambda _match: replacement,
                text,
            )
    text = _RULESPEC_ALIAS_TEMP_RE.sub("<rulespec-alias>", text)
    text = _RANDOM_TEMP_RE.sub("<tmp>", text)
    return text


def _normalized_optional_text(
    value: object,
    replacements: Mapping[str, str],
) -> str | None:
    return None if value is None else _normalize_text(value, replacements)


def _canonical_validate(
    outcome: object,
    replacements: Mapping[str, str],
) -> dict[str, object]:
    raw = outcome if isinstance(outcome, Mapping) else {}
    raw_validators = raw.get("validators", {})
    validators: dict[str, dict[str, object]] = {}
    if isinstance(raw_validators, Mapping):
        for raw_name, raw_result in sorted(
            raw_validators.items(), key=lambda item: str(item[0])
        ):
            if not isinstance(raw_result, Mapping) or raw_result.get("passed") is True:
                continue
            issues = raw_result.get("issues", [])
            normalized_issues = (
                sorted(_normalize_text(issue, replacements) for issue in issues)
                if isinstance(issues, list)
                else []
            )
            validators[str(raw_name)] = {
                "error": _normalized_optional_text(
                    raw_result.get("error"), replacements
                ),
                "issues": normalized_issues,
            }
    return {"passed": bool(raw.get("passed", False)), "validators": validators}


def _canonical_companion(
    outcome: object,
    replacements: Mapping[str, str],
) -> dict[str, object]:
    raw = outcome if isinstance(outcome, Mapping) else {}
    raw_failures = raw.get("failures", [])
    failures: list[dict[str, object]] = []
    if isinstance(raw_failures, list):
        for failure in raw_failures:
            if not isinstance(failure, Mapping):
                continue
            failures.append(
                {
                    "case": _normalized_optional_text(
                        failure.get("case"), replacements
                    ),
                    "file": _normalized_optional_text(
                        failure.get("file"), replacements
                    ),
                    "message": _normalize_text(
                        failure.get("message", "companion test failed"),
                        replacements,
                    ),
                }
            )
    failures.sort(
        key=lambda failure: json.dumps(
            failure, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    )
    cases = raw.get("cases", 0)
    if isinstance(cases, bool) or not isinstance(cases, int) or cases < 0:
        raise ValidationWaiverError("companion cases must be a non-negative integer")
    return {
        "present": bool(raw.get("present", False)),
        "passed": bool(raw.get("passed", False)),
        "path": _normalized_optional_text(raw.get("path"), replacements),
        "cases": cases,
        "failures": failures,
    }


def canonicalize_outcome(
    validate: object,
    companion: object,
    *,
    replacements: Mapping[str, str] | None = None,
) -> bytes:
    """Return deterministic semantic failure bytes, excluding volatile fields."""
    normalized_replacements = replacements or {}
    payload = {
        "schema": OUTCOME_SCHEMA,
        "validate": _canonical_validate(validate, normalized_replacements),
        "companion": _canonical_companion(companion, normalized_replacements),
    }
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def fingerprint_outcome(
    validate: object,
    companion: object,
    *,
    replacements: Mapping[str, str] | None = None,
) -> str:
    digest = hashlib.sha256(
        canonicalize_outcome(validate, companion, replacements=replacements)
    ).hexdigest()
    return f"sha256:{digest}"

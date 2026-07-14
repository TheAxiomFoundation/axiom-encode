"""Deterministic, minimal-diff updates for axiom-compose ProgramSpec scopes."""

from __future__ import annotations

import bisect
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode


class ProgramScopeError(ValueError):
    """Raised when a requested ProgramSpec scope update is unsafe or invalid."""


_SCOPE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class ProgramScopeSyncResult:
    program_spec: str
    scope: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: bool


def _mapping_value(node: MappingNode, key: str):
    matches = [
        value_node
        for key_node, value_node in node.value
        if isinstance(key_node, ScalarNode) and key_node.value == key
    ]
    if len(matches) > 1:
        raise ProgramScopeError(f"ProgramSpec contains duplicate {key!r} keys")
    return matches[0] if matches else None


def _normalize_scope_path(value: str) -> str:
    raw = str(value).strip()
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix in {".yaml", ".yml"}
    ):
        raise ProgramScopeError(
            f"scope paths must be safe repo-relative module paths without a YAML suffix: {value!r}"
        )
    return path.as_posix()


def _scope_prefix(program: str, scope: str) -> str:
    if not _SCOPE_KEY_PATTERN.fullmatch(scope):
        raise ProgramScopeError(f"scope must be a lowercase identifier: {scope!r}")
    program_path = _normalize_scope_path(program)
    jurisdiction = program_path.split("/", 1)[0]
    if scope == "federal":
        return jurisdiction.split("-", 1)[0]
    return jurisdiction


def sync_program_scope(
    *,
    repo: Path,
    program_spec: Path,
    scope: str,
    add: list[str] | tuple[str, ...] = (),
    remove: list[str] | tuple[str, ...] = (),
    write: bool = True,
) -> ProgramScopeSyncResult:
    """Apply one scope update while preserving all unrelated ProgramSpec text."""

    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise ProgramScopeError(f"repository does not exist: {repo}")
    if not _SCOPE_KEY_PATTERN.fullmatch(scope):
        raise ProgramScopeError(f"scope must be a lowercase identifier: {scope!r}")
    spec_path = Path(program_spec)
    if spec_path.is_absolute():
        try:
            spec_path = spec_path.absolute().relative_to(repo)
        except ValueError as exc:
            raise ProgramScopeError("program spec must be inside --repo") from exc
    if spec_path.suffix != ".yaml":
        raise ProgramScopeError("program spec must be a repo-relative .yaml file")
    spec_path = Path(_normalize_scope_path(spec_path.as_posix()[:-5]))
    spec_rel = Path(f"{spec_path.as_posix()}.yaml")
    lexical_spec = repo / spec_rel
    absolute_spec = lexical_spec.resolve()
    try:
        absolute_spec.relative_to(repo)
    except ValueError as exc:
        raise ProgramScopeError("program spec must resolve inside --repo") from exc
    if not absolute_spec.is_file():
        raise ProgramScopeError(f"program spec does not exist: {absolute_spec}")
    if absolute_spec != lexical_spec.absolute():
        raise ProgramScopeError("program spec path must not contain symlinks")

    text = absolute_spec.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        raise ProgramScopeError(f"invalid ProgramSpec YAML: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(root, MappingNode):
        raise ProgramScopeError("ProgramSpec root must be a mapping")
    program = payload.get("program")
    if not isinstance(program, str) or not program.strip():
        raise ProgramScopeError("ProgramSpec must declare a non-empty `program`")
    program_path = _normalize_scope_path(program.strip())
    expected_parent = Path("programs") / program_path
    if not spec_rel.is_relative_to(expected_parent):
        raise ProgramScopeError(
            f"program spec path must be nested under {expected_parent.as_posix()}/"
        )

    scope_node = _mapping_value(root, "scope")
    if not isinstance(scope_node, MappingNode):
        raise ProgramScopeError("ProgramSpec must declare a `scope` mapping")
    target_node = _mapping_value(scope_node, scope)
    if not isinstance(target_node, SequenceNode):
        raise ProgramScopeError(f"ProgramSpec scope {scope!r} must be a sequence")
    if target_node.flow_style:
        raise ProgramScopeError(
            f"ProgramSpec scope {scope!r} must use a block-style sequence"
        )
    payload_scope = payload.get("scope")
    payload_target = payload_scope.get(scope) if isinstance(payload_scope, dict) else None
    if not (
        isinstance(payload_target, list)
        and all(isinstance(item, str) for item in payload_target)
        and all(isinstance(item, ScalarNode) for item in target_node.value)
    ):
        raise ProgramScopeError(f"ProgramSpec scope {scope!r} must contain only strings")

    existing = tuple(_normalize_scope_path(item.value) for item in target_node.value)
    if len(set(existing)) != len(existing):
        raise ProgramScopeError(f"ProgramSpec scope {scope!r} contains duplicate paths")
    additions = tuple(dict.fromkeys(_normalize_scope_path(item) for item in add))
    removals = tuple(dict.fromkeys(_normalize_scope_path(item) for item in remove))
    overlap = set(additions) & set(removals)
    if overlap:
        raise ProgramScopeError(
            f"scope paths cannot be both added and removed: {sorted(overlap)}"
        )

    prefix = _scope_prefix(program.strip(), scope)
    scope_root = (repo / prefix).resolve()
    try:
        scope_root.relative_to(repo)
    except ValueError as exc:
        raise ProgramScopeError(f"scope prefix resolves outside --repo: {prefix}") from exc
    missing_modules: list[str] = []
    for item in additions:
        lexical_module = scope_root / f"{item}.yaml"
        module = lexical_module.resolve()
        try:
            module.relative_to(scope_root)
        except ValueError:
            missing_modules.append(item)
            continue
        if not module.is_file() or module != lexical_module.absolute():
            missing_modules.append(item)
    missing_modules.sort()
    if missing_modules:
        raise ProgramScopeError(
            f"scope additions do not resolve under {prefix}/: {missing_modules}"
        )

    desired_list = [item for item in existing if item not in set(removals)]
    for item in additions:
        if item in desired_list:
            continue
        if desired_list == sorted(desired_list):
            bisect.insort(desired_list, item)
        else:
            desired_list.append(item)
    desired = tuple(desired_list)
    changed = desired != existing
    if changed and write:
        indent = " " * target_node.start_mark.column
        replacement = "\n".join(f"{indent}- {item}" for item in desired)
        if not replacement:
            replacement = f"{indent}[]"
        start_index = target_node.start_mark.index - target_node.start_mark.column
        end_index = target_node.value[-1].end_mark.index
        updated = (
            text[:start_index]
            + replacement
            + text[end_index:]
        )
        try:
            updated_payload = yaml.safe_load(updated)
        except yaml.YAMLError as exc:
            raise ProgramScopeError(
                f"refusing to write invalid updated ProgramSpec YAML: {exc}"
            ) from exc
        if updated_payload.get("scope", {}).get(scope) != list(desired):
            raise ProgramScopeError(
                "refusing to write a ProgramSpec whose updated scope does not match"
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=absolute_spec.parent,
                prefix=f".{absolute_spec.name}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(updated)
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_path.chmod(absolute_spec.stat().st_mode)
            os.replace(temporary_path, absolute_spec)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return ProgramScopeSyncResult(
        program_spec=spec_rel.as_posix(),
        scope=scope,
        added=tuple(item for item in additions if item not in existing),
        removed=tuple(item for item in removals if item in existing),
        changed=changed,
    )

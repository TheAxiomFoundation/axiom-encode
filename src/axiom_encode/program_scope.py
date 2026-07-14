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

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS, RULESPEC_TEST_FILE_SUFFIX
from .repo_routing import is_composition_policy_repo_root


class ProgramScopeError(ValueError):
    """Raised when a requested ProgramSpec scope update is unsafe or invalid."""


_SCOPE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_NON_IMPORT_SCOPE_KEYS = frozenset({"exclude", "include", "jurisdictions"})


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


def _node_reference_count(root, target) -> int:
    """Count graph references to a composed node without following cycles twice."""

    references = 0
    expanded: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is target:
            references += 1
        identity = id(node)
        if identity in expanded:
            continue
        expanded.add(identity)
        if isinstance(node, MappingNode):
            for key_node, value_node in node.value:
                stack.extend((key_node, value_node))
        elif isinstance(node, SequenceNode):
            stack.extend(node.value)
    return references


def _normalize_scope_path(value: str) -> str:
    raw = str(value).strip()
    path = PurePosixPath(raw)
    if (
        not raw
        or ":" in raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix in {".yaml", ".yml"}
        or f"{path.as_posix()}.yaml".endswith(RULESPEC_TEST_FILE_SUFFIX)
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
        return "us"
    if scope == "state":
        return jurisdiction
    return scope


def _updated_scope_text(
    *,
    text: str,
    target_node: SequenceNode,
    existing: tuple[str, ...],
    desired: tuple[str, ...],
) -> str:
    """Update item lines while retaining comments and unrelated formatting."""

    indent = " " * target_node.start_mark.column
    lines = text.splitlines(keepends=True)
    sample_line = lines[target_node.start_mark.line]
    newline = "\r\n" if sample_line.endswith("\r\n") else "\n"
    if target_node.flow_style:
        if target_node.value:
            raise ProgramScopeError("non-empty scope sequences must use block style")
        line_prefix = sample_line[: target_node.start_mark.column]
        if line_prefix.strip():
            line_start = target_node.start_mark.index - target_node.start_mark.column
            line_body = sample_line.removesuffix(newline)
            line_end = line_start + len(line_body)
            key_indent = len(sample_line) - len(sample_line.lstrip(" \t"))
            item_indent = " " * (key_indent + 2)
            suffix = text[target_node.end_mark.index : line_end]
            prefix = text[: target_node.start_mark.index].rstrip(" \t")
            entries = newline.join(f"{item_indent}- {item}" for item in desired)
            return prefix + suffix + newline + entries + text[line_end:]
        replacement = f"- {desired[0]}" + "".join(
            f"{newline}{indent}- {item}" for item in desired[1:]
        )
        return (
            text[: target_node.start_mark.index]
            + replacement
            + text[target_node.end_mark.index :]
        )

    item_lines: dict[str, int] = {}
    for value, node in zip(existing, target_node.value, strict=True):
        if node.start_mark.line != node.end_mark.line:
            raise ProgramScopeError("scope entries must occupy one line each")
        item_lines[value] = node.start_mark.line

    first_item_line = min(item_lines.values())
    removed_lines = {
        line for value, line in item_lines.items() if value not in desired
    }
    insertions: dict[int, list[str]] = {}

    if not desired:
        insertions[first_item_line] = [f"{indent}  []{newline}"]
    else:
        survivors = [
            (value, item_lines[value]) for value in existing if value in desired
        ]
        new_items = [value for value in desired if value not in existing]
        for value in new_items:
            if survivors and list(existing) == sorted(existing):
                following = [line for item, line in survivors if item > value]
                anchor = following[0] if following else survivors[-1][1] + 1
                while anchor > 0 and lines[anchor - 1].lstrip().startswith("#"):
                    anchor -= 1
            elif survivors:
                anchor = survivors[-1][1] + 1
            else:
                anchor = first_item_line
            insertions.setdefault(anchor, []).append(f"{indent}- {value}{newline}")

    updated_lines: list[str] = []
    for index, line in enumerate(lines):
        pending = insertions.get(index, ())
        if pending and updated_lines and not updated_lines[-1].endswith(("\n", "\r")):
            updated_lines[-1] += newline
        updated_lines.extend(pending)
        if index not in removed_lines:
            updated_lines.append(line)
    pending = insertions.get(len(lines), ())
    if pending and updated_lines and not updated_lines[-1].endswith(("\n", "\r")):
        updated_lines[-1] += newline
    updated_lines.extend(pending)
    return "".join(updated_lines)


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

    raw_repo = Path(repo)
    if not is_composition_policy_repo_root(raw_repo):
        raise ProgramScopeError(
            f"repository must be an exact canonical rulespec-<country> checkout: {raw_repo}"
        )
    repo = raw_repo.resolve()
    if not _SCOPE_KEY_PATTERN.fullmatch(scope):
        raise ProgramScopeError(f"scope must be a lowercase identifier: {scope!r}")
    if scope in _NON_IMPORT_SCOPE_KEYS:
        raise ProgramScopeError(f"scope {scope!r} does not contain RuleSpec imports")
    spec_path = Path(program_spec)
    if spec_path.is_absolute():
        raise ProgramScopeError("program spec must be relative to --repo")
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

    with absolute_spec.open(encoding="utf-8", newline="") as source:
        text = source.read()
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
    if scope_node.flow_style:
        raise ProgramScopeError("ProgramSpec `scope` mapping must use block style")
    if _node_reference_count(root, scope_node) > 1:
        raise ProgramScopeError("ProgramSpec `scope` mapping must not use a YAML alias")
    target_node = _mapping_value(scope_node, scope)
    if not isinstance(target_node, SequenceNode):
        raise ProgramScopeError(f"ProgramSpec scope {scope!r} must be a sequence")
    if _node_reference_count(root, target_node) > 1:
        raise ProgramScopeError(
            f"ProgramSpec scope {scope!r} must not use a YAML alias"
        )
    if any(_node_reference_count(root, node) > 1 for node in target_node.value):
        raise ProgramScopeError(
            f"ProgramSpec scope {scope!r} entries must not use YAML aliases"
        )
    if target_node.flow_style and target_node.value:
        raise ProgramScopeError(
            f"non-empty ProgramSpec scope {scope!r} must use a block-style sequence"
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
    lexical_scope_root = repo / prefix
    scope_root = lexical_scope_root.resolve()
    try:
        scope_root.relative_to(repo)
    except ValueError as exc:
        raise ProgramScopeError(f"scope prefix resolves outside --repo: {prefix}") from exc
    if lexical_scope_root.is_symlink() or scope_root != lexical_scope_root.absolute():
        raise ProgramScopeError("scope root path must not contain symlinks")
    missing_modules: list[str] = []
    for item in additions:
        if PurePosixPath(item).parts[0] not in RULESPEC_ATOMIC_MODULE_ROOTS:
            missing_modules.append(item)
            continue
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

    preserve_sorted_order = list(existing) == sorted(existing)
    desired_list = [item for item in existing if item not in set(removals)]
    for item in additions:
        if item in desired_list:
            continue
        if preserve_sorted_order:
            bisect.insort(desired_list, item)
        else:
            desired_list.append(item)
    desired = tuple(desired_list)
    changed = desired != existing
    if changed and write:
        updated = _updated_scope_text(
            text=text,
            target_node=target_node,
            existing=existing,
            desired=desired,
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
                newline="",
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

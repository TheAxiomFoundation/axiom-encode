"""Safe YAML serialization that preserves strings for YAML 1.2 consumers.

PyYAML's YAML 1.1 resolver does not recognize all YAML 1.2 numbers, so its
default dumper may emit strings such as ``1e5`` or ``-0o17`` without quotes.
Keep PyYAML's safe representations, and quote strings a YAML 1.2 core-schema
consumer could otherwise resolve as numbers, booleans, or null.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

_YAML12_NON_STRING = re.compile(
    r"(?:null|Null|NULL|~|true|True|TRUE|false|False|FALSE)"
    r"|[-+]?(?:[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)"
    r"|[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF|nan|NaN|NAN)"
)


class Yaml12SafeDumper(yaml.SafeDumper):
    """SafeDumper with string typing preserved across YAML schema versions."""


def represent_yaml12_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    """Quote a string whenever a YAML 1.2 core resolver could change its type."""

    # PyYAML emits raw NEL as a YAML line break when Unicode is allowed;
    # loading that scalar normalizes NEL into a space. Double quotes escape it.
    style = (
        '"'
        if not value or _YAML12_NON_STRING.fullmatch(value) or "\x85" in value
        else None
    )
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


Yaml12SafeDumper.add_representer(str, represent_yaml12_string)


def safe_dump_yaml(data: Any, stream: Any = None, **kwargs: Any) -> Any:
    """Like ``yaml.safe_dump``, preserving string types for YAML 1.2 readers.

    This uses a local dumper subclass and never mutates PyYAML's global loader,
    dumper, or resolver configuration. Existing safe_dump keyword arguments and
    stream behavior are forwarded unchanged.
    """

    return yaml.dump(data, stream=stream, Dumper=Yaml12SafeDumper, **kwargs)


_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"
_YAML_VALUE_TAG = "tag:yaml.org,2002:value"
_DUPLICATE_KEY_REPORT_LIMIT = 10
_DUPLICATE_KEY_LABEL_LIMIT = 80
_DUPLICATE_KEY_LOCATION_LIMIT = 240
_UNCOMPARABLE_KEY = object()


def find_duplicate_mapping_key_issues(content: str) -> list[str]:
    """Report mapping keys that ``yaml.safe_load`` would silently collapse.

    Loading keeps only the last of two equal keys, so a load-and-dump rewrite
    erases the earlier entry. Keys are compared as PyYAML constructs them, with
    the equality of the loaded dict: ``rules`` and ``"rules"`` collide, and so
    do ``1``, ``0x1``, ``1.0`` and ``true``. Every mapping is checked, including
    mappings reached only through aliases, once each. Merge keys keep their own
    override semantics and are not compared.

    YAML parse errors propagate so callers can keep their parser diagnostics.
    """

    loader = _KeyMarkComposer(content)
    try:
        root = loader.get_single_node()
        if root is None:
            return []
        return _duplicate_mapping_key_issues(loader, root)
    finally:
        loader.dispose()


class _KeyMarkComposer(yaml.SafeLoader):
    """Compose with the safe schema, recording where each mapping key is written.

    An aliased key is the anchored node itself, whose mark is the anchor's, so
    the position of each key occurrence is kept per mapping entry.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.key_marks: dict[int, list[Any]] = {}

    def compose_node(self, parent: Any, index: Any) -> Any:
        if isinstance(parent, yaml.MappingNode) and index is None:
            self.key_marks.setdefault(id(parent), []).append(
                self.peek_event().start_mark
            )
        return super().compose_node(parent, index)


def _duplicate_mapping_key_issues(
    loader: _KeyMarkComposer, root: yaml.Node
) -> list[str]:
    found: list[tuple[int, int, str]] = []
    seen_nodes: set[int] = set()
    stack: list[tuple[str, yaml.Node]] = [("document root", root)]
    while stack:
        location, node = stack.pop()
        if id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        children: list[tuple[str, yaml.Node]] = []
        if isinstance(node, yaml.SequenceNode):
            children = [
                (_child_location(location, f"[{index}]"), value)
                for index, value in enumerate(node.value)
            ]
        elif isinstance(node, yaml.MappingNode):
            marks = loader.key_marks.get(id(node), [])
            first_keys: dict[Any, tuple[yaml.ScalarNode, Any]] = {}
            for entry, (key_node, value_node) in enumerate(node.value):
                if not isinstance(key_node, yaml.ScalarNode):
                    # safe_load rejects collection keys as unhashable.
                    children.append((_child_location(location, " key"), key_node))
                    children.append((_child_location(location, ".?"), value_node))
                    continue
                label = _duplicate_key_label(key_node.value)
                children.append((_child_location(location, f".{label}"), value_node))
                key = _constructed_mapping_key(loader, key_node)
                if key is _UNCOMPARABLE_KEY:
                    continue
                mark = marks[entry] if entry < len(marks) else key_node.start_mark
                try:
                    first = first_keys.get(key)
                    if first is None:
                        first_keys[key] = (key_node, mark)
                        continue
                except TypeError:
                    continue
                found.append(
                    (
                        mark.line,
                        mark.column,
                        _duplicate_key_issue(location, (key_node, mark), first),
                    )
                )
        stack.extend(reversed(children))
    found.sort(key=lambda item: (item[0], item[1]))
    issues = [issue for _line, _column, issue in found[:_DUPLICATE_KEY_REPORT_LIMIT]]
    if len(found) > _DUPLICATE_KEY_REPORT_LIMIT:
        issues.append(
            f"{len(found) - _DUPLICATE_KEY_REPORT_LIMIT} more duplicate mapping "
            "key(s) not shown."
        )
    return issues


def _constructed_mapping_key(loader: yaml.SafeLoader, key_node: yaml.ScalarNode) -> Any:
    """Construct a key the way ``safe_load`` would, or mark it uncomparable."""

    if key_node.tag == _YAML_MERGE_TAG:
        return _UNCOMPARABLE_KEY
    if key_node.tag == _YAML_VALUE_TAG:
        # PyYAML retags a plain `=` key as a string before construction.
        return key_node.value
    try:
        return loader.construct_object(key_node, deep=True)
    except (yaml.YAMLError, ValueError, TypeError, OverflowError):
        # safe_load reports a key it cannot construct with its own error.
        return _UNCOMPARABLE_KEY


def _duplicate_key_issue(
    location: str,
    duplicate: tuple[yaml.ScalarNode, Any],
    first: tuple[yaml.ScalarNode, Any],
) -> str:
    (key_node, mark), (first_node, first_mark) = duplicate, first
    label = _duplicate_key_label(key_node.value)
    position = _mark_position(mark)
    first_position = _mark_position(first_mark)
    if key_node.value == first_node.value:
        return (
            f"{location} has duplicate key `{label}` at {position} "
            f"(first at {first_position})."
        )
    return (
        f"{location} has key `{label}` at {position}, which loads as the same "
        f"key as `{_duplicate_key_label(first_node.value)}` at {first_position}."
    )


def _mark_position(mark: Any) -> str:
    return f"line {mark.line + 1}, column {mark.column + 1}"


def _duplicate_key_label(value: str) -> str:
    text = "".join(
        character if character.isprintable() else repr(character)[1:-1]
        for character in value[: _DUPLICATE_KEY_LABEL_LIMIT + 1]
    )
    if len(value) > _DUPLICATE_KEY_LABEL_LIMIT or len(text) > (
        _DUPLICATE_KEY_LABEL_LIMIT
    ):
        text = text[:_DUPLICATE_KEY_LABEL_LIMIT] + "..."
    return text


def _child_location(location: str, suffix: str) -> str:
    if location == "document root":
        child = suffix[1:] if suffix.startswith(".") else f"document root{suffix}"
    else:
        child = location + suffix
    if len(child) > _DUPLICATE_KEY_LOCATION_LIMIT:
        child = "..." + child[-(_DUPLICATE_KEY_LOCATION_LIMIT - 3) :]
    return child

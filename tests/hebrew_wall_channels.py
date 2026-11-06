"""Channel mutants for the wall: Hebrew in every rule field but the gate's (#1882).

The wall's gate reads a candidate's authoritative source text and deferral
reasons and nothing else. Review r10 reached past it through a rule's
``source:`` annotation and review r11 through a proof excerpt, so the channel
fixture puts Hebrew into every other string field of captured English and
German candidates, one field at a time and all at once, and pins main's
issue arrays for each (``tests/fixtures/non_hebrew_wall_main_issues.json``,
``channels``). The fixture's generator and the wall test build the mutants
with these same functions.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from typing import Any

# Hebrew label, pointed label, outline chain, Hebrew word, and a code point
# that is Hebrew only after NFKC (U+2135 ALEF SYMBOL folds to א).
KINDS = ("label", "pointed", "chain", "word", "nfkc")


def mutate_text(text: str, kind: str) -> str:
    if kind == "label":
        return "(א) " + text
    if kind == "pointed":
        return "(אֲ) " + text
    if kind == "chain":
        return "(ב)(1) " + text
    if kind == "word":
        return text + " חישוב"
    if kind == "nfkc":
        return text + "ℵ"
    raise ValueError(kind)


def _is_reason(path: tuple) -> bool:
    return (
        len(path) == 4
        and path[:2] == ("module", "deferred_outputs")
        and path[3] == "reason"
    )


def string_paths(value: Any, path: tuple = ()) -> Iterator[tuple]:
    """Every string leaf of a YAML value, in document order, but the reasons."""

    if isinstance(value, str):
        if not _is_reason(path):
            yield path
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from string_paths(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from string_paths(child, (*path, index))


def key_paths(value: Any, path: tuple = ()) -> Iterator[tuple]:
    """Every mapping key under a test case's ``input`` and ``output``."""

    if isinstance(value, dict):
        for key, child in value.items():
            if len(path) == 2 and path[1] in ("input", "output"):
                yield (*path, key)
            yield from key_paths(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from key_paths(child, (*path, index))


def _set(value: Any, path: tuple, new: str) -> None:
    for part in path[:-1]:
        value = value[part]
    value[path[-1]] = new


def _get(value: Any, path: tuple) -> Any:
    for part in path:
        value = value[part]
    return value


def _rename(value: Any, path: tuple, kind: str) -> None:
    parent = _get(value, path[:-1])
    items = list(parent.items())
    parent.clear()
    for key, child in items:
        parent[mutate_text(str(key), kind) if key == path[-1] else key] = child


def mutants(payload: dict, test_cases: list) -> Iterator[tuple[str, dict, list]]:
    """``(label, payload, test cases)`` for each channel mutant of one base.

    Each string field takes the kind its position selects; a proof excerpt
    or its citation path takes every kind; each test-case key is renamed;
    and five mutants put one kind into every field but ``format`` at once.
    Keeping the format valid lets those combined mutants reach the analyzer;
    each individual format mutant still exercises the shape refusal.
    """

    fields = [("payload", path) for path in string_paths(payload)]
    fields += [("tests", path) for path in string_paths(test_cases)]
    for index, (where, path) in enumerate(fields):
        excerpt = "atoms" in path and path[-1] in ("excerpt", "corpus_citation_path")
        kinds = KINDS if excerpt else (KINDS[index % len(KINDS)],)
        for kind in kinds:
            new_payload = copy.deepcopy(payload)
            new_tests = copy.deepcopy(test_cases)
            target = new_payload if where == "payload" else new_tests
            _set(target, path, mutate_text(_get(target, path), kind))
            yield (
                f"{where}:{json.dumps(path, ensure_ascii=True)}:{kind}",
                new_payload,
                new_tests,
            )
    for index, path in enumerate(key_paths(test_cases)):
        kind = KINDS[index % len(KINDS)]
        new_tests = copy.deepcopy(test_cases)
        _rename(new_tests, path, kind)
        yield (
            f"key:{json.dumps(path, ensure_ascii=True)}:{kind}",
            copy.deepcopy(payload),
            new_tests,
        )
    for kind in KINDS:
        new_payload = copy.deepcopy(payload)
        new_tests = copy.deepcopy(test_cases)
        for path in string_paths(new_payload):
            if path == ("format",):
                continue
            _set(new_payload, path, mutate_text(_get(new_payload, path), kind))
        for path in string_paths(new_tests):
            _set(new_tests, path, mutate_text(_get(new_tests, path), kind))
        yield f"all:{kind}", new_payload, new_tests

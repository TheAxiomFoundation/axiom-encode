"""In-process memos for repository-wide RuleSpec scans.

Several checks read every module under a RuleSpec checkout on every call: the
canonical-concept index built while a prompt is prepared, and the executable
index that the duplicate-upstream check builds while a candidate is
validated. Most of that time is pure-Python YAML parsing of files whose bytes
have not changed since the previous scan in the same process.

This module keeps only values that are a pure function of a file's exact text
(its "text facts"). The key is the text itself, not a path, a size or a
modification time: a hit needs an equal string, so a changed file can never
be served a stale value. Paths, repository identity and every filesystem
check stay outside the memo and run on every call.

One ``yaml.safe_load`` of a text serves every registered fact function, so a
module parsed by one scan is not parsed again by another scan in the same
process. The fact functions only read the parsed payload; none keeps or
returns any part of it except immutable values.

The parse is PyYAML's pure-Python ``yaml.safe_load``, the loader every scan
used before. libyaml (``yaml.CSafeLoader``) is not used: it accepts text that
``yaml.safe_load`` rejects (for example ``"a:\\tb"`` or ``"x: |#-"``), and a
caller cannot tell those cases apart without the pure-Python parse.

Nothing is written to disk and nothing outlives the process. When a fact
function raises, nothing is stored for it and the caller runs its unmemoized
code, which raises the same exception from the same place.

A value is stored only when the parse succeeded and the parsed payload nests
at most ``_MAX_MEMOIZED_NESTING`` levels. The parse and the fact functions
recurse once per level, so a deeper text can raise ``RecursionError`` when the
caller's stack is deep and succeed when it is shallow; a stored value would
hide that difference. A parse that failed is not stored either, because the
depth it reached before the error is not known. Such texts are computed again
on every call, at the caller's own stack depth.

``AXIOM_ENCODE_SCAN_CACHE=0`` (also ``off``, ``false`` or ``no``) turns every
scan memo off, so the scans run the code they ran before the memos existed.
Any value other than ``1``, ``on``, ``true``, ``yes`` or empty also turns them
off. The trusted signing supervisor builds its child environment from empty
and drops the variable, so there the memos are always on.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, NamedTuple

import yaml

SCAN_CACHE_ENVIRONMENT_VARIABLE = "AXIOM_ENCODE_SCAN_CACHE"

_ON_VALUES = frozenset({"", "1", "on", "true", "yes"})

# Keys (texts) plus stored values held per fact memo, counted in characters.
_DEFAULT_MAX_CHARS = 128 * 1024 * 1024

# Deepest parsed nesting that is memoized (see the module docstring).
_MAX_MEMOIZED_NESTING = 100

_CONTAINER_TYPES = (dict, list, tuple, set, frozenset)

_MISSING = object()


def scan_cache_enabled() -> bool:
    """Return whether the scan memos are on (read on every call)."""

    value = os.environ.get(SCAN_CACHE_ENVIRONMENT_VARIABLE)
    if value is None:
        return True
    return value.strip().lower() in _ON_VALUES


class ParseOutcome(NamedTuple):
    """``yaml.safe_load(text)``: its value, or the ``Exception`` it raised."""

    payload: Any
    error: Exception | None


class TextFactsUnavailable(Exception):
    """No memoized value: the caller must run its unmemoized code."""


FactFunction = Callable[[str, ParseOutcome], Any]


def _nesting_exceeds(payload: Any, limit: int) -> bool:
    """Return whether ``payload`` nests more than ``limit`` containers deep.

    Walks one level at a time without recursion. A container reached twice on
    one level (a YAML alias) is visited once; a cyclic payload exceeds every
    limit.
    """

    level: list[Any] = [payload] if isinstance(payload, _CONTAINER_TYPES) else []
    depth = 0
    while level:
        depth += 1
        if depth > limit:
            return True
        children: dict[int, Any] = {}
        for container in level:
            if isinstance(container, dict):
                items: Any = (*container.keys(), *container.values())
            else:
                items = container
            for item in items:
                if isinstance(item, _CONTAINER_TYPES):
                    children[id(item)] = item
        level = list(children.values())
    return False


def _value_chars(value: Any) -> int:
    """Return the approximate size of one stored value, in characters.

    Strings and bytes count their length; every value, container and
    dataclass field also counts one.
    """

    total = 0
    pending = [value]
    while pending:
        current = pending.pop()
        total += 1
        if isinstance(current, (str, bytes)):
            total += len(current)
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            pending.extend(current)
        elif dataclasses.is_dataclass(current) and not isinstance(current, type):
            pending.extend(
                getattr(current, field.name) for field in dataclasses.fields(current)
            )
    return total


class _FactMemo:
    """Least-recently-used map from an exact text to one immutable value.

    ``chars`` counts each entry's text and its stored value, so a small text
    with a large value cannot grow the memo past ``max_chars``.
    """

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.chars = 0
        self.entries: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, content: str) -> Any:
        entry = self.entries.get(content)
        if entry is None:
            return _MISSING
        self.entries.move_to_end(content)
        return entry[0]

    def put(self, content: str, value: Any) -> None:
        if content in self.entries:
            return
        cost = len(content) + _value_chars(value)
        if cost > self.max_chars:
            return
        self.entries[content] = (value, cost)
        self.chars += cost
        while self.chars > self.max_chars and self.entries:
            _, (_, evicted_cost) = self.entries.popitem(last=False)
            self.chars -= evicted_cost


_LOCK = threading.Lock()
_FACT_FUNCTIONS: dict[str, FactFunction] = {}
_FACT_MEMOS: dict[str, _FactMemo] = {}


def register_text_facts(
    name: str,
    function: FactFunction,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> None:
    """Register a pure, read-only function of ``(text, parse outcome)``.

    The function must return an immutable value (strings, tuples, frozen
    dataclasses, ``None``) and must not change the parsed payload, because the
    same payload is handed to every registered function.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    with _LOCK:
        if name in _FACT_FUNCTIONS and _FACT_FUNCTIONS[name] is not function:
            raise ValueError(f"text facts {name!r} are already registered")
        _FACT_FUNCTIONS[name] = function
        _FACT_MEMOS.setdefault(name, _FactMemo(max_chars))


def text_facts(content: str, name: str) -> Any:
    """Return ``function(content, yaml.safe_load outcome)`` for ``name``.

    Raises ``TextFactsUnavailable`` when the memos are off, when ``content``
    is not exactly a ``str``, or when the function raised; the caller then runs
    its unmemoized code. The value is computed but not stored when the parse
    failed or the payload nests too deep (see the module docstring).
    """

    if type(content) is not str or not scan_cache_enabled():
        raise TextFactsUnavailable(name)
    with _LOCK:
        memo = _FACT_MEMOS[name]
        cached = memo.get(content)
        if cached is not _MISSING:
            memo.hits += 1
            return cached
        functions = dict(_FACT_FUNCTIONS)
    try:
        outcome = ParseOutcome(yaml.safe_load(content), None)
    except Exception as exc:  # noqa: BLE001 - a fact function decides
        outcome = ParseOutcome(None, exc)
    computed: dict[str, Any] = {}
    for fact_name, function in functions.items():
        try:
            computed[fact_name] = function(content, outcome)
        except Exception:  # noqa: BLE001 - the caller's own code decides
            continue
    storable = outcome.error is None and not _nesting_exceeds(
        outcome.payload, _MAX_MEMOIZED_NESTING
    )
    with _LOCK:
        _FACT_MEMOS[name].misses += 1
        if storable:
            for fact_name, value in computed.items():
                _FACT_MEMOS[fact_name].put(content, value)
    if name not in computed:
        raise TextFactsUnavailable(name)
    return computed[name]


def scan_memo_stats() -> dict[str, dict[str, int]]:
    """Return entry, hit and miss counts per fact memo."""

    with _LOCK:
        return {
            name: {
                "entries": len(memo.entries),
                "chars": memo.chars,
                "hits": memo.hits,
                "misses": memo.misses,
            }
            for name, memo in _FACT_MEMOS.items()
        }


def clear_scan_memos() -> None:
    """Drop every stored value (tests use this to start cold)."""

    with _LOCK:
        for memo in _FACT_MEMOS.values():
            memo.entries.clear()
            memo.chars = 0
            memo.hits = 0
            memo.misses = 0

"""Conservative recognition of outlines whose labels occupy separate lines.

The walk authenticates a whole outline before exposing any marker offsets. A
partial walk is unsafe: a skipped duplicate, a table cell, or a broken sequence
could make a later number appear to be a subsection label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BareLabel:
    token: str
    path: tuple[str, ...]
    start: int
    marker_end: int
    end: int


_NUMBER = re.compile(r"([1-9]\d*)([A-Z]?)\Z")
_LETTER = re.compile(r"[a-z]{1,3}\Z")
_ROMAN = re.compile(
    r"(?=[ivxlcdm]+\Z)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})\Z"
)
_TABLE = re.compile(r"\b(?:table|tabular|columns?|rows?|grid)\b", re.IGNORECASE)
_PARENTHETICAL = re.compile(r"^\s*\((?:[1-9]\d*[A-Z]?|[a-z]{1,3})\)(?:\s|$)")
_LIST_INTRO = re.compile(r"[:—–-](?:\s+(?:and|or))?\s*$")
_SENTENCE_END = re.compile(r"[.;:—–-](?:\s+(?:and|or))?\s*$")
_PREDICATE = re.compile(
    r"\b(?:is|are|must|shall|may|has|have|applies|apply|means|includes|"
    r"overrides|does|do|was|were|can|cannot|will|ends|ceases|continues|"
    r"requires|entitles|prohibits|permits|provides)\b"
)
_DEFINITION = re.compile(
    r"^[ \t]*(?P<term>[a-z][a-z -]*?)"
    r"(?:[ \t]*,[^\n|]*?)?[ \t]+means\b",
    re.MULTILINE,
)
_ROW_CAPTION = re.compile(
    r"^(?:number|count|amount|rate|percentage|year|annual|weekly|monthly|"
    r"first|second|third|fourth|fifth|total|subtotal)\b",
    re.IGNORECASE,
)


def _roman_value(token: str) -> int | None:
    if not _ROMAN.fullmatch(token):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for letter in reversed(token):
        value = values[letter]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total


def _ordinal(token: str, kind: str) -> tuple[int, int] | None:
    if kind == "number":
        match = _NUMBER.fullmatch(token)
        if match:
            return int(match[1]), ord(match[2]) - 64 if match[2] else 0
    elif kind == "roman":
        value = _roman_value(token)
        if value is not None:
            return value, 0
    elif _LETTER.fullmatch(token):
        value = 0
        for letter in token:
            value = value * 26 + ord(letter) - 96
        return value, 0
    return None


def _successor(
    previous: str, current: str, kind: str, *, numbered_letters: bool = False
) -> bool:
    earlier = _ordinal(previous, kind)
    later = _ordinal(current, kind)
    if earlier is None or later is None:
        return False
    if later == (earlier[0] + 1, 0):
        return True
    if kind == "number":
        return later == (earlier[0], earlier[1] + 1)
    if kind == "letter" and not numbered_letters:
        # NZ insertions retain the original sibling's label: c, ca, cb, d.
        # Also admit ordinary multi-letter numbering: ... z, aa, ab, ... .
        return current == previous + "a" or (
            len(previous) > 1
            and current == chr(ord(previous[0]) + 1)
            and previous[0] != "z"
        )
    return False


def _is_token(line: str) -> bool:
    return bool(
        _NUMBER.fullmatch(line) or _LETTER.fullmatch(line) or _ROMAN.fullmatch(line)
    )


def bare_outline(text: str) -> tuple[BareLabel, ...] | None:
    """Return one positively ordered outline, otherwise retain all obligations.

    All token-shaped lines participate in the walk, including lines whose
    follower cannot authenticate a label. Never silently skip one. Explicit
    table evidence, mixed parenthetical layouts, gaps, and duplicate labels
    reject the entire interpretation. Numeric-only prose lists lack evidence
    distinguishing subsection labels from count/table entries and are retained.
    """
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    if any(
        "\t" in line or "|" in line or _TABLE.search(line) or _PARENTHETICAL.match(line)
        for line in lines
    ):
        return None
    candidates = [index for index, line in enumerate(lines) if _is_token(line.strip())]
    if not candidates:
        return None
    first = lines[candidates[0]].strip()
    root_kind = "number" if _NUMBER.fullmatch(first) else "letter"
    kinds = (
        ("number", "letter", "roman") if root_kind == "number" else ("letter", "roman")
    )
    stack: list[str] = []
    numbered_letters: list[bool] = []
    labels: list[BareLabel] = []
    for candidate_number, index in enumerate(candidates):
        token = lines[index].strip()
        if index + 1 >= len(lines):
            return None
        following = lines[index + 1].strip()
        if (
            not following
            or _is_token(following)
            or not re.search(r"[A-Za-z]", following)
        ):
            return None
        before = lines[index - 1].strip() if index else ""
        options = [
            level
            for level in range(len(stack) - 1, -1, -1)
            if _successor(
                stack[level],
                token,
                kinds[level],
                numbered_letters=numbered_letters[level],
            )
        ]
        child_level = len(stack)
        if (
            child_level < len(kinds)
            and _ordinal(token, kinds[child_level]) == (1, 0)
            and (not stack or _LIST_INTRO.search(before))
        ):
            options.append(child_level)
        # Competing sibling/descent readings are an ambiguity, not a choice.
        if len(options) != 1:
            return None
        depth = options[0]
        if depth == 0 and root_kind == "number":
            # Numeric labels must head complete prose rather than table/list
            # captions; a number embedded in a sentence cannot start a sibling.
            # Intervening unlabelled lines could introduce a count or table, so
            # they cannot authenticate a numeric sibling even if it is ordered.
            if (
                not _SENTENCE_END.search(following)
                or len(re.findall(r"[A-Za-z]+", following)) < 3
                or not _PREDICATE.search(following)
                or _ROW_CAPTION.match(following)
                or (index and not _SENTENCE_END.search(before))
                or (
                    candidate_number
                    and index - 1 != candidates[candidate_number - 1] + 1
                )
            ):
                return None
        elif not stack and index and not _LIST_INTRO.search(before):
            return None
        numbering = depth < len(stack) and (
            numbered_letters[depth] or (stack[depth] == "z" and token == "aa")
        )
        stack = stack[:depth] + [token]
        numbered_letters = numbered_letters[:depth] + [numbering]
        labels.append(
            BareLabel(
                token,
                tuple(stack),
                starts[index],
                starts[index] + len(lines[index].rstrip("\r\n")),
                len(text),
            )
        )
    if len(labels) < 2:
        return None
    if root_kind == "number" and not any(len(label.path) > 1 for label in labels):
        return None
    result: list[BareLabel] = []
    definition_starts = (
        [start for start, _term in bare_definition_headers(text, tuple(labels))]
        if root_kind == "letter"
        else []
    )
    # A subsequent definition is an unlabelled sibling of the enclosing
    # definition, rather than continuation text of its last paragraph.
    for index, label in enumerate(labels):
        boundaries = [
            later.start
            for later in labels[index + 1 :]
            if len(later.path) <= len(label.path)
        ]
        boundaries.extend(
            start for start in definition_starts if start > label.marker_end
        )
        result.append(replace(label, end=min(boundaries, default=len(text))))
    return tuple(result)


def bare_definition_headers(
    text: str, outline: tuple[BareLabel, ...]
) -> tuple[tuple[int, str], ...]:
    """Unlabelled definition heads in a positively authenticated outline.

    A marker's following text belongs to that marker even if it says ``means``.
    Definitions never invent additional statutory citation fragments.
    """
    if not outline:
        return ()
    labelled_text_starts = {text.find("\n", label.marker_end) + 1 for label in outline}
    return tuple(
        (match.start(), match.group("term").strip())
        for match in _DEFINITION.finditer(text)
        if match.start() not in labelled_text_starts
    )


def mask_bare_labels(text: str, outline: tuple[BareLabel, ...] | None = None) -> str:
    """Hide only authenticated marker characters, preserving source offsets."""
    outline = bare_outline(text) if outline is None else outline
    if outline is None:
        return text
    for label in reversed(outline):
        text = (
            text[: label.start]
            + " " * (label.marker_end - label.start)
            + text[label.marker_end :]
        )
    return text


def slice_bare_path(text: str, fragments: tuple[str, ...]) -> str | None:
    outline = bare_outline(text)
    if outline is None:
        return None
    matches = [label for label in outline if label.path == fragments]
    if len(matches) != 1:
        return None
    label = matches[0]
    return text[label.start : label.end].strip()

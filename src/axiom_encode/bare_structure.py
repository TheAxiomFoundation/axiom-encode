"""Conservative recognition of outlines whose labels occupy separate lines."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class BareLabel:
    token: str
    path: tuple[str, ...]
    start: int
    marker_end: int
    end: int


_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx")
_TOKEN = re.compile(r"(?:[1-9]\d*[A-Z]?|[a-z]{1,3})\Z")


def _ordinal(token: str, kind: str) -> tuple[int, int] | None:
    if kind == "number":
        match = re.fullmatch(r"([1-9]\d*)([A-Z]?)", token)
        if match:
            return int(match[1]), ord(match[2]) - 64 if match[2] else 0
    elif kind == "roman":
        if token in _ROMAN:
            return _ROMAN.index(token) + 1, 0
    elif re.fullmatch(r"[a-z]{1,2}", token):
        value = 0
        for letter in token:
            value = value * 26 + ord(letter) - 96
        return value, 0
    return None


def _successor(previous: tuple[int, int], current: tuple[int, int]) -> bool:
    return current == (previous[0] + 1, 0) or current == (previous[0], previous[1] + 1)


def bare_outline(text: str) -> tuple[BareLabel, ...] | None:
    """Authenticate a complete ordered outline, or retain every source obligation.

    Isolated numbers, tabular rows, duplicate labels, and broken sequences are
    deliberately unrecognized. A numeric root needs siblings or letter children.
    """
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    candidates: list[tuple[str, int, int, str]] = []
    for index, line in enumerate(lines):
        token = line.strip()
        if not _TOKEN.fullmatch(token):
            continue
        following = index + 1
        if following >= len(lines):
            continue
        body = lines[following].strip()
        # Text must immediately follow; a table cell or another label cannot
        # authenticate a number. Tabs and pipes are explicit table evidence.
        if (not body or _TOKEN.fullmatch(body) or "\t" in lines[following]
                or "|" in body or len(re.findall(r"[A-Za-z]+", body)) < 3):
            continue
        # Complete prose, rather than a short table heading, is required.
        if not re.search(r"[.;:—–-]$", body):
            continue
        before = lines[index - 1].strip() if index else ""
        candidates.append((token, starts[index], starts[index] + len(line.rstrip("\r\n")), before))
    if not candidates:
        return None
    root_kind = "number" if candidates[0][0][0].isdigit() else "letter"
    kinds = ("number", "letter", "roman") if root_kind == "number" else ("letter", "roman")
    stack: list[tuple[str, tuple[int, int]]] = []
    labels: list[BareLabel] = []
    for token, start, marker_end, before in candidates:
        # Select a successor at an active level before considering descent.
        depth = None
        for level in range(len(stack) - 1, -1, -1):
            ordinal = _ordinal(token, kinds[level])
            if ordinal is not None and _successor(stack[level][1], ordinal):
                depth = level
                break
        if depth is None:
            level = len(stack)
            if level < len(kinds) and _ordinal(token, kinds[level]) == (1, 0):
                # A nested list must be introduced explicitly by its parent.
                if not stack or re.search(r"[:;—–-]$", before):
                    depth = level
        if depth is None:
            return None
        ordinal = _ordinal(token, kinds[depth])
        assert ordinal is not None
        stack = stack[:depth] + [(token, ordinal)]
        labels.append(BareLabel(token, tuple(item[0] for item in stack), start, marker_end, len(text)))
    if len(labels) < 2:
        return None
    # A lone numeric root with no child is never enough evidence.
    if root_kind == "number" and len({label.path[0] for label in labels}) == 1 and not any(len(label.path) > 1 for label in labels):
        return None
    result = []
    for index, label in enumerate(labels):
        end = next((later.start for later in labels[index + 1:] if len(later.path) <= len(label.path)), len(text))
        result.append(BareLabel(label.token, label.path, label.start, label.marker_end, end))
    return tuple(result)


def mask_bare_labels(text: str, outline: tuple[BareLabel, ...] | None = None) -> str:
    """Hide only authenticated marker characters, preserving source offsets."""
    outline = bare_outline(text) if outline is None else outline
    if outline is None:
        return text
    for label in reversed(outline):
        text = text[:label.start] + " " * (label.marker_end - label.start) + text[label.marker_end:]
    return text


def slice_bare_path(text: str, fragments: tuple[str, ...]) -> str | None:
    outline = bare_outline(text)
    if outline is None:
        return None
    matches = [label for label in outline if label.path == fragments]
    if len(matches) != 1:
        return None
    label = matches[0]
    return text[label.start:label.end].strip()

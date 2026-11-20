"""Structural reading of a source unit's bullet lists.

``complete-source-unit:source-explicit-conditions/gates-collapsed`` reads a
condition clause from flat text. This module gives it the list structure of
the scanned unit text instead: which physical blocks are a list chapeau, its
marked items, notes and unmarked children, and where the list ends.

It is a closed reader. A list is an *alternatives* list only when its chapeau
has one of two exact forms (see ``_ANY_CUE_CHAPEAU`` and
``_ANY_SUBJECT_CHAPEAU``) and its layout is one this reader can state without
a guess. Every other list is ``unrecognised``, and the caller then keeps the
flat reading (every bullet under a cue is one more simultaneous fact). There
is no list of words that switch a reading off.

The reading is used only while ``STRUCTURAL_GATE_READING_ENABLED`` is true. The
default is false: the validator then never calls into this module.
See ``docs/gates-structural-reading.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The one switch. While it is False the gates-collapsed check is the flat
# reader, byte for byte.
STRUCTURAL_GATE_READING_ENABLED = False

_MARKERS = "●•"
_BLOCK_LIMIT = 2000
_ITEM_LIMIT = 12

# The quantifier that says "one item is enough". Closed: six exact phrases.
_ANY_QUANTIFIER = (
    r"(?:any|one\s+or\s+more|either|any\s+one|at\s+least\s+one|one)"
    r"\s+of\s+the\s+following"
)
# What may stand between the quantifier and the colon. Closed and positive: an
# optional plain noun, then an optional affirmative verb. A negated tail
# ("do not apply") or any other word is not this form.
_ANY_TAIL = (
    r"(?:\s+(?:conditions?|situations?|circumstances?|criteria|events?|reasons?))?"
    r"(?:\s+(?:occur|occurs|apply|applies|exist|exists|"
    r"is\s+met|are\s+met|is\s+true|are\s+true))?"
)
# "... when|if <quantifier> <tail>:" at the very end of the chapeau. The cue
# stands directly before the quantifier, so no negator or second quantifier
# ("each one of", "not any of") can come between them.
_ANY_CUE_CHAPEAU = re.compile(
    rf"\b(?P<cue>when|if)\s+{_ANY_QUANTIFIER}{_ANY_TAIL}\s*:\s*\Z",
    re.IGNORECASE,
)
# "Any of the following <one clause>:" as the whole chapeau: the items are the
# grammatical subject, one sentence, no inner sentence punctuation.
_ANY_SUBJECT_CHAPEAU = re.compile(r"Any\s+of\s+the\s+following\b[^:.;!?]*:\s*\Z")
# Explicit conjunctive evidence. Read the same way as an unrecognised list; the
# label only documents that the wording itself says "all".
_ALL_CHAPEAU = re.compile(
    r"\b(?:all|each|every\s+one|both)\s+of\s+the\s+following\b[^.;!?]*:\s*\Z",
    re.IGNORECASE,
)
# The flat reader ends a proposition after sentence punctuation only before
# one of these starts. The end of the last item of an alternatives list is
# given the same rule, nothing wider.
_SENTENCE_START = re.compile(r"\s+(?:[A-Z(]|\d+\s*[.)]|\d+\s+(?:After|For)\b)")


@dataclass(frozen=True)
class SourceBlock:
    """One physical block of the unit text: ``text[start:end]``."""

    start: int
    end: int
    kind: str  # "paragraph", "item", "note" or "example"


@dataclass(frozen=True)
class SourceList:
    """A chapeau block and the marked items under it."""

    chapeau: SourceBlock
    items: tuple[SourceBlock, ...]
    # End of the last marked item, with its unmarked children when it has any.
    end: int
    # False when unmarked text after the last item may still belong to it.
    end_is_certain: bool
    # Marked items that are themselves followed by unmarked children.
    items_with_children: frozenset[int]
    reading: str  # "alternatives", "conjunctive" or "unrecognised"
    # The chapeau states its own condition cue directly before the quantifier.
    cue_form: bool

    def tail_starts_sentence(self, text: str) -> bool:
        """Say whether the text after the list starts a new proposition."""

        return (
            self.end_is_certain and _SENTENCE_START.match(text, self.end) is not None
            # The flat reader's lookahead, anchored at the end of the list.
        )


def _blocks(text: str) -> tuple[SourceBlock, ...] | None:
    """Split the text into physical blocks, or decline an unreadable layout."""

    blocks: list[SourceBlock] = []
    offset = 0
    previous_blank = True
    for line in text.split("\n"):
        end = offset + len(line)
        stripped = line.strip()
        if "\r" in line or "\t" in line:
            return None
        if not stripped:
            previous_blank = True
        else:
            if line[0] in _MARKERS:
                if len(line) < 3 or line[1] != " " or line[2] == " ":
                    return None
                kind = "item"
            elif line[0] == " " or stripped[0] in _MARKERS:
                # Indented text or an indented marker: nesting by indent is
                # not a layout this reader states. Decline the whole unit.
                return None
            elif re.match(r"NOTE [A-Z]", line):
                kind = "note"
            elif re.match(r"Example(?:s)?\b", line):
                kind = "example"
            else:
                kind = "paragraph"
            if previous_blank or kind != "paragraph" or not blocks:
                blocks.append(SourceBlock(offset, end, kind))
            else:
                # A wrapped line continues the block before it.
                last = blocks[-1]
                blocks[-1] = SourceBlock(last.start, end, last.kind)
            previous_blank = False
            if len(blocks) > _BLOCK_LIMIT:
                return None
        offset = end + 1
    return tuple(blocks)


def _ends_with_colon(text: str, block: SourceBlock) -> bool:
    return text[block.start : block.end].rstrip().endswith(":")


def _chapeau_reading(chapeau_text: str) -> tuple[str, bool]:
    collapsed = " ".join(chapeau_text.split())
    if len(re.findall(r"\bfollowing\b", collapsed, re.IGNORECASE)) != 1:
        return "unrecognised", False
    if _ANY_CUE_CHAPEAU.search(collapsed) is not None:
        return "alternatives", True
    if _ANY_SUBJECT_CHAPEAU.fullmatch(collapsed) is not None:
        return "alternatives", False
    if _ALL_CHAPEAU.search(collapsed) is not None:
        return "conjunctive", False
    return "unrecognised", False


def parse_source_lists(text: str) -> tuple[SourceList, ...]:
    """Return the lists whose chapeau is an unmarked top-level paragraph.

    A list is: a paragraph that ends with a colon, directly followed by marked
    items with one marker. Between items the reader accepts ``NOTE`` and
    ``Example`` blocks (commentary on the item before them) and, after an item
    that ends with a colon, unmarked paragraphs (that item's own children; the
    corpus prints nested items without a marker). The list ends at the first
    unmarked paragraph after an item that does not end with a colon.

    A list is ``unrecognised`` (flat reading) when a marked item ends with a
    colon and has no unmarked child: its marked children would then look like
    siblings, and the reader cannot tell them apart.
    """

    blocks = _blocks(text)
    if blocks is None:
        return ()
    lists: list[SourceList] = []
    index = 0
    while index < len(blocks):
        chapeau = blocks[index]
        if (
            chapeau.kind != "paragraph"
            or not _ends_with_colon(text, chapeau)
            or index + 1 >= len(blocks)
            or blocks[index + 1].kind != "item"
        ):
            index += 1
            continue
        marker = text[blocks[index + 1].start]
        items: list[SourceBlock] = []
        with_children: set[int] = set()
        readable = True
        end = chapeau.end
        cursor = index + 1
        while cursor < len(blocks):
            block = blocks[cursor]
            if block.kind == "item":
                if text[block.start] != marker:
                    readable = False
                items.append(block)
                end = block.end
            elif block.kind in {"note", "example"}:
                # Commentary on the item before it; the list goes on only when
                # another marked item follows.
                if not any(later.kind == "item" for later in blocks[cursor + 1 :]):
                    break
                following = next(
                    later
                    for later in blocks[cursor + 1 :]
                    if later.kind not in {"note", "example"}
                )
                if following.kind != "item":
                    break
            elif _ends_with_colon(text, items[-1]) or (len(items) - 1) in with_children:
                with_children.add(len(items) - 1)
                end = block.end
            else:
                break
            cursor += 1
        for position, item in enumerate(items):
            if _ends_with_colon(text, item) and position not in with_children:
                readable = False
        if len(items) > _ITEM_LIMIT:
            readable = False
        last = len(items) - 1
        reading, cue_form = _chapeau_reading(text[chapeau.start : chapeau.end])
        if not readable and reading == "alternatives":
            reading, cue_form = "unrecognised", False
        lists.append(
            SourceList(
                chapeau=chapeau,
                items=tuple(items),
                end=end,
                end_is_certain=last not in with_children
                and not _ends_with_colon(text, items[last]),
                items_with_children=frozenset(with_children),
                reading=reading,
                cue_form=cue_form,
            )
        )
        # The next chapeau can only be a paragraph after this list.
        index = cursor
    return tuple(lists)


def any_cue_chapeau_prefix(chapeau_text: str) -> str | None:
    """Return the chapeau without its "when|if <quantifier> ...:" ending."""

    match = _ANY_CUE_CHAPEAU.search(chapeau_text)
    return None if match is None else chapeau_text[: match.start()]


def item_body(text: str, item: SourceBlock) -> str:
    """Return the words of a marked item, without its marker."""

    return text[item.start + 2 : item.end].strip()

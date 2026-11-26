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
# The chapeau forms that put the list's items under a condition cue of the
# chapeau itself, as joint conditions. Only these forms, and the cue form of
# alternatives, bind a rule that quotes items to the whole list. A missed
# form leaves the flat behaviour. In "When X, all of the following apply:"
# the items are consequences of X, so that form binds nothing; a count form
# ("at least 2 of the following") binds nothing, because no single rule path
# holds every item.
_CONDITION_LIST_CHAPEAU = re.compile(
    r"\b(?:when|if|provided\s+that)\s+(?:"
    r"(?:all|each|both|every\s+one|all\s+(?:two|three|four|five|six))"
    r"(?:\s+of)?\s+the\s+following" + _ANY_TAIL + r"|"
    r"the\s+following\s+(?:conditions|requirements|criteria)\s+"
    r"(?:are\s+met|are\s+satisfied|apply|are\s+true)"
    r"|(?:[A-Za-z]+\s+){1,4}?(?:meets?|satisfy|satisfies)\s+"
    r"(?:all|each)\s+of\s+the\s+following"
    r"(?:\s+(?:conditions|requirements|criteria))?"
    r")\s*:\s*\Z|\bif\s*:\s*\Z",
    re.IGNORECASE,
)
# Wording that can turn the sense of a chapeau or of an item around, or that
# adds a qualifier. It is a second layer: the positive requirements are the
# closed forms. A chapeau or an item that holds one of these words is not
# verified as alternatives; the flat reading stays.
_REVERSING_WORD = re.compile(
    r"(?:\b(?:not|no|never|none|neither|nor|except\w*|unless|without|but|cannot|"
    r"fail\w*|den(?:y|ied|ies|ial)|refus\w*|prohibit\w*|exclud\w*|ineligib\w*|"
    r"invalid\w*|disallow\w*|disqualif\w*|barred|void\w*|ceas\w*|los[et]s?|"
    r"terminat\w*|suspend\w*|revok\w*|cancel\w*|reject\w*|non\w*|longer|"
    r"rarely|hardly|seldom|scarcely|barely|withh[eo]ld\w*|forfeit\w*|laps\w*|"
    r"withdr[ae]w\w*|expir\w*|"
    r"instead|rather|other\s+than)\b|n['’]t\b)",
    re.IGNORECASE,
)
_QUALIFYING_WORD = re.compile(
    r"\b(?:if|when|whenever|where|while|until|once|provided|providing|only|both|"
    r"either|as|who|whom|whose|that|which|also|plus|together|additionally|"
    r"including|includes?|whether|because|since|after|before|during|then|with|"
    r"within|more|less|least|most|than|over|under|above|below|exceed\w*|even|"
    r"each|every|all|any)\b",
    re.IGNORECASE,
)
# A glossary mark "(g)" or a cross-reference "(See ...)" at the end of an
# item. Any other parenthetical stays in the item and is read with it.
_TRAILING_REFERENCE = re.compile(r"(?:\s*\((?:[a-z]|See [^()]*)\))+\s*\.?\s*\Z")
# A second condition or quantifier before the cue of a cue-form chapeau.
_STEM_CUE = re.compile(
    r"\b(?:if|when|whenever|where|while|until|unless|whether|even|provided|"
    r"providing|each|every|both|either|neither|asks?|decides?)\b",
    re.IGNORECASE,
)
# A cue-form chapeau states an affirmative main clause before its cue.
_AFFIRMATIVE_VERB = re.compile(r"\b(?:must|may|is|are|can|shall|will)\b", re.IGNORECASE)


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
    # Marked items that are themselves followed by unmarked children.
    items_with_children: frozenset[int]
    reading: str  # "alternatives", "conjunctive" or "unrecognised"
    # The chapeau states its own condition cue directly before the quantifier.
    cue_form: bool
    # A rule that quotes an item is read against the whole list.
    binds_items: bool
    # The chapeau is a plain affirmative and the list ends the unit text: the
    # list can be verified as alternatives. Text after the list could add to
    # its condition, and no closed form tells that it does not.
    verifiable: bool = False


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


def chapeau_is_affirmative(chapeau_text: str, *, cue_form: bool) -> bool:
    """Say whether a chapeau of alternatives states a plain affirmative."""

    collapsed = " ".join(chapeau_text.split())
    cue = _ANY_CUE_CHAPEAU.search(collapsed)
    stem = collapsed[: cue.start()] if cue_form and cue is not None else collapsed
    if cue_form:
        if cue is None or _AFFIRMATIVE_VERB.search(stem) is None:
            return False
        stem = re.sub(r"\bonly\s*\Z", "", stem, flags=re.IGNORECASE)
        if _STEM_CUE.search(stem) is not None:
            return False
    return _REVERSING_WORD.search(stem) is None


def item_is_plain(body: str) -> bool:
    """Say whether an item is one plain alternative that can be verified.

    After an optional leading "When" or "If" and without trailing
    parentheticals, the item holds words, at most one ", and" joint, no word
    that reverses its sense and no word that opens a qualifier.
    """

    words = re.sub(r"\A(?:when|if)\s+", "", body.strip(), flags=re.IGNORECASE)
    words = _TRAILING_REFERENCE.sub("", words).rstrip(" .")
    return (
        re.fullmatch(
            r"[A-Za-z0-9$][A-Za-z0-9$ '’-]*(?:, and [A-Za-z0-9$ '’-]+)?", words
        )
        is not None
        and re.match(r"[A-Z0-9$]", body.strip()) is not None
        and re.search(r"\b(?:and|or|nor)\s*\Z", words, re.IGNORECASE) is None
        and len(re.findall(r"\bor\b", words, re.IGNORECASE)) <= 1
        and _REVERSING_WORD.search(words) is None
        and _QUALIFYING_WORD.search(words) is None
    )


def item_words(body: str) -> str:
    """Return an item without a leading cue and trailing references."""

    words = re.sub(r"\A(?:when|if)\s+", "", body.strip(), flags=re.IGNORECASE)
    return _TRAILING_REFERENCE.sub("", words).rstrip(" .")


def item_or_sides(body: str) -> tuple[str, str] | None:
    """Return the two sides of the one "or" of a plain item, if it has one."""

    words = re.sub(r"\A(?:when|if)\s+", "", body.strip(), flags=re.IGNORECASE)
    words = _TRAILING_REFERENCE.sub("", words).rstrip(" .")
    parts = re.split(r"\bor\b", words, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    # "or" joins two parallel phrases: the right side, and as many words at
    # the end of the left side. Words before them (a shared subject) belong
    # to both cases.
    right = parts[1].split()
    left = parts[0].split()[-len(right) :] if right else []
    return (" ".join(left), " ".join(right)) if left and right else None


def _chapeau_reading(chapeau_text: str) -> tuple[str, bool, bool]:
    """Return the reading, the cue form and whether item proofs bind."""

    collapsed = " ".join(chapeau_text.split())
    if len(re.findall(r"\bfollowing\b", collapsed, re.IGNORECASE)) != 1:
        bare_if = re.search(r"\bif\s*:\s*\Z", collapsed, re.IGNORECASE) is not None
        return "unrecognised", False, bare_if and "following" not in collapsed.lower()
    binds = _CONDITION_LIST_CHAPEAU.search(collapsed) is not None
    if _ANY_CUE_CHAPEAU.search(collapsed) is not None:
        return "alternatives", True, True
    if _ANY_SUBJECT_CHAPEAU.fullmatch(collapsed) is not None:
        return "alternatives", False, False
    if _ALL_CHAPEAU.search(collapsed) is not None:
        return "conjunctive", False, binds
    return "unrecognised", False, binds


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
                following = next(
                    (
                        later
                        for later in blocks[cursor + 1 :]
                        if later.kind not in {"note", "example"}
                    ),
                    None,
                )
                if following is None or following.kind != "item":
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
        reading, cue_form, binds_items = _chapeau_reading(
            text[chapeau.start : chapeau.end]
        )
        if not readable:
            # The items cannot be told apart from nested ones: flat reading,
            # and no rule is bound to a list whose members are not known.
            reading, cue_form, binds_items = "unrecognised", False, False
        lists.append(
            SourceList(
                chapeau=chapeau,
                items=tuple(items),
                end=end,
                items_with_children=frozenset(with_children),
                reading=reading,
                cue_form=cue_form,
                binds_items=binds_items,
                verifiable=reading == "alternatives"
                and not text[end:].strip()
                and chapeau_is_affirmative(
                    text[chapeau.start : chapeau.end], cue_form=cue_form
                ),
            )
        )
        # The next chapeau can only be a paragraph after this list.
        index = cursor
    owned = {item.start for found in lists for item in found.items}
    if any(block.kind == "item" and block.start not in owned for block in blocks):
        # A marked item outside every list: the list it belongs to was
        # interrupted (a page line, a heading), so no list here is whole.
        return ()
    return tuple(lists)


def any_cue_chapeau_prefix(chapeau_text: str) -> str | None:
    """Return the chapeau without its "when|if <quantifier> ...:" ending."""

    match = _ANY_CUE_CHAPEAU.search(chapeau_text)
    return None if match is None else chapeau_text[: match.start()]


def item_body(text: str, item: SourceBlock) -> str:
    """Return the words of a marked item, without its marker."""

    return text[item.start + 2 : item.end].strip()

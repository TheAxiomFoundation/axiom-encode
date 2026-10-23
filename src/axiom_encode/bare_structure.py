"""Conservative recognition of outlines whose labels occupy separate lines.

The walk authenticates a whole outline before exposing any marker offsets. A
partial walk is unsafe: a skipped duplicate, a table cell, or a broken sequence
could make a later number appear to be a subsection label.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BareLabel:
    token: str
    path: tuple[str, ...]
    start: int
    marker_end: int
    end: int


_NUMBER = re.compile(r"([1-9]\d*)([A-Z]?)\Z")
_NUMERIC_TOKEN = re.compile(r"\d[\dA-Za-z]*\Z")
_LETTER = re.compile(r"[a-z]+\Z")
_ROMAN = re.compile(
    r"(?=[ivxlcdm]+\Z)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})\Z"
)
_TABLE = re.compile(r"\b(?:table|tabular|columns?|rows?|grid)\b", re.IGNORECASE)
_PARENTHETICAL = re.compile(r"^\s*\((?:[1-9]\d*[A-Z]?|[a-z]+)\)(?:\s|$)")
_LIST_INTRO = re.compile(r"[:—–-](?:\s+(?:and|or))?\s*$")
_SENTENCE_END = re.compile(r"[.;:—–-](?:\s+(?:and|or))?\s*$")
_PREDICATE = re.compile(
    r"\b(?:is|are|must|shall|may|has|have|applies|apply|means|includes|"
    r"overrides|does|do|was|were|can|cannot|will|ends|ceases|continues|"
    r"requires|entitles|prohibits|permits|provides)\b"
)
_NUMERIC_SUBJECT_INTRO = re.compile(
    r"^(?:a|an|the|another|this|these|those|he|she|it|they|you|we)\s+",
    re.IGNORECASE,
)
_NUMERIC_LOCAL_PREDICATE = re.compile(_PREDICATE.pattern, re.IGNORECASE)
_TEMPORAL_UNIT = (
    r"(?:seconds?|minutes?|hours?|days?|weeks?|fortnights?|months?|quarters?|"
    r"years?|decades?|centur(?:y|ies)|millenni(?:um|a))"
)
_NUMERIC_TEMPORAL_FREQUENCY = re.compile(
    r"^(?:a|an|the|another|this|these|those)\s+(?:(?:whole|entire)\s+)?"
    + _TEMPORAL_UNIT
    + r"\b",
    re.IGNORECASE,
)
_NUMERIC_DISTRIBUTIVE_POSSESSION = re.compile(
    r"^each\s+(?P<subject>[A-Za-z][A-Za-z ’'-]{0,159}?)\s+has\s+"
    r"(?:a|an|the)\s+\S",
    re.IGNORECASE,
)
_NUMERIC_CONDITIONAL_INTRO = re.compile(
    r"^(?:if|unless|where|when|whenever|while|whether)\s+", re.IGNORECASE
)
_NUMERIC_DISCOURSE_INTRO = re.compile(
    r"^(?:however|otherwise|accordingly|therefore)(?:,\s+|\s+)", re.IGNORECASE
)
_NUMERIC_SCOPE_INTRO = re.compile(r"^in\s+this\s+section,\s+", re.IGNORECASE)
_NUMERIC_NAMED_MAIN_CLAUSE = re.compile(
    r"^[A-Z]{2,16}\s+(?:must|shall|may|can|cannot|will)\s+"
)
_NUMERIC_LEGAL_SUBJECT = re.compile(
    r"^(?:sections?|subsections?|paragraphs?|clauses?|subclauses?|parts?|"
    r"schedules?|chapters?|regulations?|articles?)\s+"
    r"[1-9]\d*[A-Z]*(?:\([A-Za-z0-9]+\))*"
    r"(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or|to)\s+)"
    r"[1-9]\d*[A-Z]*(?:\([A-Za-z0-9]+\))*)*\s+",
    re.IGNORECASE,
)
_NUMERIC_RELATIVE_SUBJECT = re.compile(
    r"\b(?:which|that|who|whom|whose|where|when)\b", re.IGNORECASE
)
_DEFINITION = re.compile(
    r"^[ \t]*(?P<term>[^\W\d_](?:[^\W_]|[ \t'’()/-])*?)"
    r"(?:[ \t]*,[^\n|]*?)?[ \t]+means\b",
    re.MULTILINE,
)
_DEFINITION_HEADER = re.compile(r"^[ \t]*(?P<term>[^|\r\n]+?)[ \t]+means\b")
_ROW_CAPTION = re.compile(
    r"^(?:number|count|amount|rate|percentage|year|annual|weekly|monthly|"
    r"first|second|third|fourth|fifth|total|subtotal)\b",
    re.IGNORECASE,
)
_NUMERIC_CONTINUATION = re.compile(
    r"^(?:%|(?:" + _TEMPORAL_UNIT + r"|child(?:ren)?|persons?|people|members?|"
    r"depend[ae]nts?|percent|per\s+cent|dollars?|cents?|or\s+more)\b)",
    re.IGNORECASE,
)
_VALUE_INTRO = re.compile(
    r"\b(?:is|are|equals?|totals?|(?:as )?follows|as shown|"
    r"(?:given|set out|specified|prescribed|listed) below|the following|"
    r"number|count|amount|rate|percentage|year|total|value)"
    r"[.;:—–-]?\s*$",
    re.IGNORECASE,
)
_VALUE_CONTEXT = re.compile(
    r"\b(?:lists?|entries|following|below|listed|shown|"
    r"(?:this|that|as|so)\s+many)\b",
    re.IGNORECASE,
)
_QUANTITY_NOUN = (
    r"(?:counts?|amounts?|rates?|percentages?|numbers?|years?|"
    r"totals?|values?|figures?|columns?|rows?|quantit(?:y|ies)|"
    r"limits?|ceilings?|caps?|thresholds?|maximums?|maxima|minimums?|minima)"
)
_UNRESOLVED_QUANTITY = re.compile(rf"\b{_QUANTITY_NOUN}\b", re.IGNORECASE)
_QUANTITY_ASSIGNMENT = re.compile(
    _QUANTITY_NOUN + r"(?:\s+of\s+[A-Za-z ’'-]{1,120})?"
    r"\s+(?:is|are|equals?|totals?)\s+\$?\d+(?:\.\d+)?(?:%|\b)",
    re.IGNORECASE,
)
_QUANTITY_COMPLEMENT = re.compile(
    r"\b(?:at|by|with|for|on)\s+(?:a|an|the)\s+$", re.IGNORECASE
)
_QUANTITY_PRESCRIBED = re.compile(
    r"\s+(?:prescribed|specified|approved|provided|payable)\b", re.IGNORECASE
)
_LINE_ENDINGS = "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_VARIABLE_INTRO = re.compile(
    r"\b(?:where|formula|symbols?|variables?)\b", re.IGNORECASE
)
_VARIABLE_MEANING = re.compile(r"^(?:is|means|represents|denotes|equals)\b")
_VARIABLE_EXPRESSION = re.compile(r"\b[a-zA-Z]\s*(?:[+*/÷×=]|−|\s-\s)")
_REFERENCE_INTRO = re.compile(
    r"\b(?:sections?|subsections?|paragraphs?|subparagraphs?|clauses?|"
    r"subclauses?|parts?|schedules?|chapters?|regulations?|articles?|see)\s*$",
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
        if match and len(match[1]) <= 9:
            return int(match[1]), ord(match[2]) - 64 if match[2] else 0
    elif kind == "roman":
        value = _roman_value(token)
        if value is not None:
            return value, 0
    elif len(token) <= 32 and _LETTER.fullmatch(token):
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
        _NUMERIC_TOKEN.fullmatch(line)
        or _LETTER.fullmatch(line)
        or _ROMAN.fullmatch(line)
    )


def _numeric_main_clause_has_subject(text: str) -> bool:
    """Use only the first comma-bounded clause's own subject and predicate."""
    clause = text.split(",", 1)[0]
    if _NUMERIC_TEMPORAL_FREQUENCY.match(clause):
        return False
    subject = _NUMERIC_SUBJECT_INTRO.match(clause)
    if subject is None:
        return False
    predicate = _NUMERIC_LOCAL_PREDICATE.search(clause, subject.end())
    if predicate is None or _NUMERIC_RELATIVE_SUBJECT.search(
        clause, subject.end(), predicate.start()
    ):
        return False
    introduction = subject.group().strip().casefold()
    if introduction in {"a", "an", "the", "another"} and not re.search(
        r"[^\W\d_]", clause[subject.end() : predicate.start()]
    ):
        return False
    return not (
        introduction in {"a", "an"}
        and predicate.group().casefold() in {"are", "were", "have"}
    )


def _numeric_follower_is_independent_prose(text: str) -> bool:
    """Require a subject, not a quantity's noun or relative continuation."""
    legal_subject = _NUMERIC_LEGAL_SUBJECT.match(text)
    if legal_subject is not None:
        return bool(_NUMERIC_LOCAL_PREDICATE.match(text, legal_subject.end()))
    possession = _NUMERIC_DISTRIBUTIVE_POSSESSION.match(text)
    if possession is not None:
        subject = possession.group("subject")
        return not (
            _PREDICATE.search(subject.lower())
            or _NUMERIC_RELATIVE_SUBJECT.search(subject)
        )
    if _NUMERIC_SUBJECT_INTRO.match(text):
        return _numeric_main_clause_has_subject(text)
    discourse = _NUMERIC_DISCOURSE_INTRO.match(text)
    if discourse is not None:
        return _numeric_main_clause_has_subject(text[discourse.end() :])
    scope = _NUMERIC_SCOPE_INTRO.match(text)
    if scope is not None:
        remainder = text[scope.end() :]
        return bool(
            _numeric_main_clause_has_subject(remainder) or _DEFINITION.match(remainder)
        )
    if _NUMERIC_CONDITIONAL_INTRO.match(text):
        # A conditional fragment can qualify an operative count. Only an
        # independently headed clause after the final comma supplies evidence
        # of a new subsection. Earlier commas may introduce appositives inside
        # the condition; unsupported complex main clauses retain obligations.
        _condition, comma, remainder = text.rpartition(",")
        if comma and remainder[:1].isspace():
            remainder = remainder.lstrip()
            return bool(
                _numeric_main_clause_has_subject(remainder)
                or _NUMERIC_NAMED_MAIN_CLAUSE.match(remainder)
            )
    return False


def _quantity_is_resolved(text: str, quantity: re.Match[str]) -> bool:
    """A prescribed rate in a prepositional complement is not a pending value."""
    return bool(
        _QUANTITY_ASSIGNMENT.match(text, quantity.start())
        or re.match(r"\s+of\s+\$?\d+(?:\.\d+)?(?:%|\b)", text[quantity.end() :])
        or (
            _QUANTITY_COMPLEMENT.search(text[: quantity.start()])
            and _QUANTITY_PRESCRIBED.match(text, quantity.end())
        )
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
    candidates = [index for index, line in enumerate(lines) if _is_token(line.strip())]
    if not candidates:
        return None
    if any(
        _NUMERIC_TOKEN.fullmatch(token := lines[index].strip())
        and _NUMBER.fullmatch(token) is None
        for index in candidates
    ):
        # Unknown numeric spellings may name a sibling rather than continue
        # the preceding paragraph. Retain them without inventing legal paths.
        return None
    if any(
        "\t" in line or "|" in line or _TABLE.search(line) or _PARENTHETICAL.match(line)
        for line in lines
    ):
        return None
    for index in candidates:
        if index:
            if _REFERENCE_INTRO.search(lines[index - 1].strip()):
                # A wrapped citation can imitate the next sibling ordinal.
                # Its referenced label remains running text, not a child unit.
                return None
            # Formula variables can look exactly like an alphabetic outline.
            # Their symbols are operands, rather than statutory child paths.
            nearby_expression = any(
                _VARIABLE_EXPRESSION.search(line)
                for line in lines[max(0, index - 4) : index]
            )
            if (
                index + 1 < len(lines)
                and _VARIABLE_MEANING.match(lines[index + 1].strip())
                and _VARIABLE_INTRO.search(lines[index - 1].strip())
            ) or nearby_expression:
                return None
    first = lines[candidates[0]].strip()
    root_kind = (
        "number" if _NUMBER.fullmatch(first) else "roman" if first == "i" else "letter"
    )
    kinds = {
        "number": ("number", "letter", "roman"),
        "letter": ("letter", "roman"),
        "roman": ("roman",),
    }[root_kind]
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
                # A noun-headed follower can complete an operative count,
                # even with a predicate and a child list of its own. Require
                # positive independent prose, rather than enumerating nouns.
                or not _numeric_follower_is_independent_prose(following)
                or _ROW_CAPTION.match(following)
                # A unit/count follower can complete an operative quantity,
                # even when the remainder is a full predicate sentence.
                or _NUMERIC_CONTINUATION.match(following)
                or (index and not _SENTENCE_END.search(before))
                # A dangling introduction may be completed by an operative
                # count. It cannot authenticate a new numeric subsection.
                or (index and _LIST_INTRO.search(before))
                or (
                    index
                    and (
                        _VALUE_INTRO.search(before)
                        or _VALUE_CONTEXT.search(before)
                        or any(
                            not _quantity_is_resolved(before, quantity)
                            for quantity in _UNRESOLVED_QUANTITY.finditer(before)
                        )
                        or _ROW_CAPTION.match(before)
                    )
                )
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
                starts[index] + len(lines[index].rstrip(_LINE_ENDINGS)),
                len(text),
            )
        )
    if len(labels) < 2:
        return None
    if root_kind == "number" and not any(len(label.path) > 1 for label in labels):
        # Ordered numbers and complete prose do not establish legal identity.
        # Retain base obligations for detached numeric-only lists; explicit
        # parenthesized subsections belong to the ordinary marker parser.
        return None
    labelled_text_lines = {index + 1 for index in candidates}
    for index, line in enumerate(lines):
        header = _DEFINITION_HEADER.match(line)
        if (
            index not in labelled_text_lines
            and header is not None
            and re.search(r"[^\W\d_]", header.group("term"))
            and _DEFINITION.match(line) is None
        ):
            # Do not attach an unknown detached definition to the preceding
            # paragraph. Exotic term spellings retain the legacy obligations.
            return None
    result: list[BareLabel] = []
    definition_starts = [
        start for start, _term in bare_definition_headers(text, tuple(labels))
    ]
    introduced_definitions = {
        start
        for start in definition_starts
        if (line_index := bisect_right(starts, start) - 1) > 0
        and _LIST_INTRO.search(lines[line_index - 1].strip())
    }
    root_starts = {label.path: label.start for label in labels if len(label.path) == 1}
    # A subsequent definition is an unlabelled sibling of the enclosing
    # definition, rather than continuation text of its last paragraph.
    next_at_depth = [len(text)] * len(kinds)
    for label in reversed(labels):
        depth = len(label.path) - 1
        end = min(next_at_depth[: depth + 1])
        definition_index = bisect_right(definition_starts, label.marker_end)
        # A detached definition can end the preceding definition's child
        # list. Numeric subsection ancestors retain the later definition.
        # Require a preceding detached head in the same enclosing source
        # unit; a definition inside a labelled paragraph remains its text.
        parent_start = (
            root_starts[label.path[:1]]
            if root_kind == "number" and len(label.path) > 1
            else 0
        )
        prior_definition = bisect_right(definition_starts, label.start) - 1
        if (
            (root_kind != "number" or len(label.path) > 1)
            and definition_index < len(definition_starts)
            and prior_definition >= 0
            and definition_starts[prior_definition] >= parent_start
        ):
            if (
                definition_starts[definition_index] < end
                and definition_starts[definition_index] in introduced_definitions
            ):
                # The preceding child may explicitly introduce this head.
                # Do not move a possibly nested definition to its ancestor.
                return None
            end = min(end, definition_starts[definition_index])
        result.append(replace(label, end=end))
        next_at_depth[depth] = label.start
    ordered = tuple(reversed(result))
    by_path = {label.path: label for label in ordered}
    for label in ordered:
        for depth in range(1, len(label.path)):
            ancestor = by_path[label.path[:depth]]
            if not ancestor.start <= label.start < label.end <= ancestor.end:
                # A detached definition cannot break an ancestor's list and
                # then leave subsequent tokens attached to that ancestor.
                return None
    return ordered


def bare_definition_headers(
    text: str, outline: tuple[BareLabel, ...]
) -> tuple[tuple[int, str], ...]:
    """Unlabelled definition heads in a positively authenticated outline.

    A marker's following text belongs to that marker even if it says ``means``.
    Definitions never invent additional statutory citation fragments.
    """
    if not outline:
        return ()
    label_starts = {label.start for label in outline}
    labelled_text_starts: set[int] = set()
    definitions: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if offset in label_starts:
            labelled_text_starts.add(offset + len(line))
        match = _DEFINITION.match(line)
        if match is not None and offset not in labelled_text_starts:
            definitions.append((offset + match.start(), match.group("term").strip()))
        offset += len(line)
    return tuple(definitions)


def mask_bare_labels(text: str, outline: tuple[BareLabel, ...] | None = None) -> str:
    """Hide only authenticated marker characters, preserving source offsets."""
    outline = bare_outline(text) if outline is None else outline
    if outline is None:
        return text
    pieces: list[str] = []
    offset = 0
    for label in outline:
        pieces.extend(
            (text[offset : label.start], " " * (label.marker_end - label.start))
        )
        offset = label.marker_end
    pieces.append(text[offset:])
    return "".join(pieces)


def slice_bare_path(
    text: str,
    fragments: tuple[str, ...],
    *,
    outline: tuple[BareLabel, ...] | None = None,
) -> str | None:
    outline = bare_outline(text) if outline is None else outline
    if outline is None:
        return None
    matches = [label for label in outline if label.path == fragments]
    if len(matches) != 1:
        return None
    label = matches[0]
    # The selected marker names the source unit itself. Leaving it in a leaf
    # would let downstream unstructured excerpts glue it onto operative text;
    # a selected numeric marker would also look like an operative number.
    raw = text[label.marker_end : label.end]
    start = label.marker_end + len(raw) - len(raw.lstrip())
    sliced = raw.strip()
    descendants = tuple(
        child
        for child in outline
        if child.path[: len(fragments)] == fragments
        and len(child.path) > len(fragments)
    )
    if any(child.start >= label.end or child.end > label.end for child in descendants):
        return None
    if not descendants:
        return sliced
    projected = bare_outline(sliced)
    expected = tuple(
        (
            child.path[len(fragments) :],
            child.start - start,
            child.marker_end - start,
            min(child.end - start, len(sliced)),
        )
        for child in descendants
    )
    if (
        projected is not None
        and tuple(
            (child.path, child.start, child.marker_end, child.end)
            for child in projected
        )
        == expected
    ):
        return sliced
    # A singleton descendant cannot independently authenticate its sequence.
    # Render only already proven descendant markers in the explicit layout.
    # Detached definitions have boundaries that the parenthetical recognizer
    # does not preserve, so those projections must fail before generation.
    if any(
        start <= definition_start < label.end
        for definition_start, _term in bare_definition_headers(text, outline)
    ):
        return None
    pieces: list[str] = []
    offset = start
    for child in descendants:
        pieces.extend((text[offset : child.start], f"({child.token})"))
        offset = child.marker_end
    pieces.append(text[offset : label.end])
    return "".join(pieces).strip()

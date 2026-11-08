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
    # False for a root that only the ordered walk, not the text, places: a
    # root after the highest subsection the section cites. Its outline
    # position stands, but its number stays a source obligation.
    proven: bool = True


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
_NUMERIC_RELATIVE_SUBJECT = re.compile(
    r"\b(?:which|that|who|whom|whose|where|when)\b", re.IGNORECASE
)
# A follower headed by cited legal units, such as `Sections 219 and 436 do not
# apply ...`. It is admitted only inside a self-cited complete section.
_NUMERIC_LEGAL_SUBJECT = re.compile(
    r"^(?:sections?|subsections?|paragraphs?|clauses?|subclauses?|parts?|"
    r"schedules?|chapters?|regulations?|articles?)\s+"
    r"[1-9]\d*[A-Z]*(?:\([A-Za-z0-9]+\))*"
    r"(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or|to)\s+)"
    r"[1-9]\d*[A-Z]*(?:\([A-Za-z0-9]+\))*)*\s+",
    re.IGNORECASE,
)
# The statutory body of exactly one section: `<jurisdiction>/statute/.../
# section/<id>`, with an ordinary section number (`220`, `32a`, `dg-18`), no
# subsection fragment or anchor, and no schedule, appendix, annex, table,
# form, repealed or nested-section segment. Only such a body must contain
# every subsection it cites.
_COMPLETE_SECTION_PATH = re.compile(
    r"[a-z]{2}(?:-[a-z]{2})?/statute/"
    r"(?:(?![a-z0-9-]*(?:sch|appendix|annex|table|form|repeal|section))"
    r"[a-z0-9-]+/)*"
    r"section/(?:[a-z]{1,3}-)?[0-9]+[A-Za-z]{0,3}"
)
_SUBSECTION_REFERENCE = re.compile(
    r"\bsubsections?\s+(?P<labels>\([1-9]\d*[A-Z]?\)"
    r"(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or|to)\s+)\([1-9]\d*[A-Z]?\))*)",
    re.IGNORECASE,
)
_REFERENCE_LABEL = re.compile(r"\((?P<label>[1-9]\d*[A-Z]?)\)")
# A reference qualified by another unit (`subsection (2) of section 71`) does
# not name a unit of this section.
_QUALIFIED_REFERENCE_SUFFIX = re.compile(r"\s*(?:of|in)\b", re.IGNORECASE)
# Words that could make a citation address another unit, quote or amend text,
# or describe the numbers as values. Each form is listed, so each is tested.
# Table words are absent: main's walk already rejects any text naming them.
_FOREIGN_REFERENCE_WORDS = (
    # Other units and instruments.
    "s", "ss", "sch", "section", "sections", "act", "acts", "code", "codes",
    "title", "titles", "statute", "statutes", "law", "laws", "ordinance",
    "ordinances", "order", "orders", "decree", "decrees", "instrument",
    "instruments", "provision", "provisions", "schedule", "schedules", "part",
    "parts", "subpart", "subparts", "division", "divisions", "subdivision",
    "subdivisions", "chapter", "chapters", "article", "articles", "paragraph",
    "paragraphs", "subparagraph", "subparagraphs", "clause", "clauses",
    "subclause", "subclauses", "item", "items", "rule", "rules", "regulation",
    "regulations",
    # Amendment verbs.
    "replace", "replaces", "replaced", "insert", "inserts", "inserted",
    "repeal", "repeals", "repealed", "omit", "omits", "omitted", "substitute",
    "substitutes", "substituted", "amend", "amends", "amended", "amendment",
    "amendments", "following",
    # Value words.
    "size", "sizes", "figure", "figures", "number", "numbers", "numeral",
    "numerals", "count", "counts", "persons", "people",
)  # fmt: skip
_FOREIGN_REFERENCE_SYMBOLS = '"“”‘’«»§¶'
_FOREIGN_REFERENCE_CONTEXT = re.compile(
    r"\b(?:"
    + "|".join(_FOREIGN_REFERENCE_WORDS)
    + r")\b|["
    + re.escape(_FOREIGN_REFERENCE_SYMBOLS)
    + "]",
    re.IGNORECASE,
)
# Any other parenthesized spelling of a root, glued, spaced or full-width.
_ROOT_MARKER = r"[(（]\s*({})\s*[)）]"


def _own_subsection_references(
    text: str, root_spans: list[tuple[int, int]]
) -> tuple[set[str], bool]:
    """Labels this text cites as its own subsections, and whether all are clean.

    The second value is false when any unqualified subsection reference sits in
    a root unit, children included, whose text could address another unit,
    amend or quote text, or call the numbers values; such text proves nothing.
    """
    labels: set[str] = set()
    root_starts = [start for start, _end in root_spans]
    clean_units: set[int] = set()
    for reference in _SUBSECTION_REFERENCE.finditer(text):
        if _QUALIFIED_REFERENCE_SUFFIX.match(text, reference.end()):
            continue
        index = bisect_right(root_starts, reference.start()) - 1
        if index < 0:
            return set(), False
        if index not in clean_units:
            # A reference holds none of these words itself, so one search of
            # the whole unit covers every reference inside it.
            if _FOREIGN_REFERENCE_CONTEXT.search(text, *root_spans[index]):
                return set(), False
            clean_units.add(index)
        labels.update(
            label["label"]
            for label in _REFERENCE_LABEL.finditer(reference.group("labels"))
        )
    return labels, True


def _self_cited_numeric_roots(
    text: str, labels: list[BareLabel], *, first_line_is_label: bool
) -> set[str] | None:
    """The bare roots a complete section's text cites as its own subsections.

    A complete section that cites its own `subsection (N)` contains that
    subsection. When no other `(N)` marker exists and the walk is a single
    ordered outline from the first line, the bare root `N` is the only unit
    that can be subsection N, and the roots before it are subsections 1 to
    N - 1. Nothing proves the roots after the highest cited one, so callers
    keep their numbers. Returns ``None`` when the text proves nothing.

    Repealed subsections: the NZ corpus drops a repealed subsection from its
    section's text entirely, with no label line and no tombstone. A repeal
    before a live subsection therefore leaves a gap in the bare roots
    (`1`, `3`, `4`), and the ordered walk rejects the gap before this proof
    runs, so it proves nothing and callers keep main's numbers. The proof can
    go wrong only if a stray bare number fills the gap exactly. Bodies the
    corpus resolver composes from descendant rows also drop repealed rows,
    but they join chunks with blank lines, which the walk rejects too.
    """
    if not first_line_is_label:
        return None
    roots = [label for label in labels if len(label.path) == 1]
    root_spans = [
        (
            root.marker_end,
            roots[index + 1].start if index + 1 < len(roots) else len(text),
        )
        for index, root in enumerate(roots)
    ]
    cited, clean = _own_subsection_references(text, root_spans)
    tokens = {root.token for root in roots}
    if not clean or not cited or not cited <= tokens:
        return None
    # Positions of the label parentheses inside subsection references; any
    # other spelling of a root marker could be the cited subsection itself.
    cited_markers = {
        reference.start("labels") + label.start()
        for reference in _SUBSECTION_REFERENCE.finditer(text)
        for label in _REFERENCE_LABEL.finditer(reference.group("labels"))
    }
    # One pass for every root token keeps the scan linear in the text.
    root_marker = re.compile(
        _ROOT_MARKER.format("|".join(map(re.escape, sorted(tokens, key=len)[::-1])))
    )
    if not all(
        marker.start() in cited_markers for marker in root_marker.finditer(text)
    ):
        return None
    return cited


def _legal_subject_follower(text: str) -> bool:
    subject = _NUMERIC_LEGAL_SUBJECT.match(text)
    return bool(subject and _NUMERIC_LOCAL_PREDICATE.match(text, subject.end()))


def is_complete_section_path(corpus_citation_path: str) -> bool:
    return bool(_COMPLETE_SECTION_PATH.fullmatch(corpus_citation_path or ""))


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


def bare_outline(
    text: str, *, corpus_citation_path: str = ""
) -> tuple[BareLabel, ...] | None:
    """Return one positively ordered outline, otherwise retain all obligations.

    All token-shaped lines participate in the walk, including lines whose
    follower cannot authenticate a label. Never silently skip one. Explicit
    table evidence, mixed parenthetical layouts, gaps, and duplicate labels
    reject the entire interpretation. Numeric-only prose lists lack evidence
    distinguishing subsection labels from count/table entries and are retained.

    One exception needs positive evidence from the text itself. When
    ``corpus_citation_path`` names one complete section and the section cites
    its own ``subsection (N)`` for a bare root N (see
    ``_self_cited_numeric_roots``), a root follower may be headed by cited
    legal units. Every other check holds, including the nested-label one.
    The citation proves only the cited roots and the roots before them: a
    later root, and everything under it, is returned with ``proven=False``,
    so ``mask_bare_labels`` keeps its number and ``slice_bare_path`` refuses
    it.
    """
    section_scope = is_complete_section_path(corpus_citation_path)
    legal_subject_roots = False
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
            independent = _numeric_follower_is_independent_prose(following)
            # Provisional: the self-citation proof below must still hold.
            cited_unit_subject = (
                not independent and section_scope and _legal_subject_follower(following)
            )
            legal_subject_roots = legal_subject_roots or cited_unit_subject
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
                or not (independent or cited_unit_subject)
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
                            not _QUANTITY_ASSIGNMENT.match(before, quantity.start())
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
        # Ordered numbers and complete prose do not establish legal identity,
        # and neither does a self-citation alone: a numeric-only list keeps
        # base's obligations.
        return None
    cited: set[str] | None = None
    if legal_subject_roots:
        cited = (
            _self_cited_numeric_roots(
                text, labels, first_line_is_label=candidates[0] == 0
            )
            if section_scope
            else None
        )
        if cited is None:
            # Only the complete section's own subsection citation proves that
            # a legal-unit-headed follower opens a subsection.
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
    if cited is not None:
        # The citation proves the cited roots and the ones before them. A root
        # after the highest cited one may still be a count key, so it and its
        # children keep their numbers: only positive evidence discharges one.
        highest = max(_ordinal(token, "number") for token in cited)
        unproven = {
            label.path
            for label in ordered
            if len(label.path) == 1 and _ordinal(label.token, "number") > highest
        }
        ordered = tuple(
            replace(label, proven=False) if label.path[:1] in unproven else label
            for label in ordered
        )
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


def mask_bare_labels(
    text: str,
    outline: tuple[BareLabel, ...] | None = None,
    *,
    corpus_citation_path: str = "",
) -> str:
    """Hide only authenticated marker characters, preserving source offsets."""
    outline = (
        bare_outline(text, corpus_citation_path=corpus_citation_path)
        if outline is None
        else outline
    )
    if outline is None:
        return text
    pieces: list[str] = []
    offset = 0
    for label in outline:
        if not label.proven:
            continue
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
    corpus_citation_path: str = "",
) -> str | None:
    outline = (
        bare_outline(text, corpus_citation_path=corpus_citation_path)
        if outline is None
        else outline
    )
    if outline is None:
        return None
    matches = [label for label in outline if label.path == fragments]
    if len(matches) != 1:
        return None
    label = matches[0]
    if not label.proven:
        # Nothing in the text proves this unit is the subsection it is asked
        # for, so it cannot be sliced out as one.
        return None
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

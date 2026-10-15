"""Finite syntactic ancestry for authoritative bibliographic editorial notices.

This only confirms a local omission; it never enumerates required source branches."""

import hashlib
import re
from dataclasses import dataclass

BIBLIOGRAPHIC_NOTICE = re.compile(
    r"\((?P<label>[1-9]\d*)\)\s*(?:Repealed|Omitted)\.\s+"
    r"Pub\.\s*L\.\s*\d+[-–]\d+,\s*"
    r"(?:div\.\s*[A-Z],\s*)?title\s+[IVXLCDM]+,\s*"
    r"(?:§|section|sec\.)\s*\d+(?:\([A-Za-z0-9]+\))*\s*,\s*"
    r"(?:Jan\.|January|Feb\.|February|Mar\.|March|Apr\.|April|May|"
    r"June|July|Aug\.|August|Sept\.|September|Oct\.|October|Nov\.|November|"
    r"Dec\.|December)\s+(?:[1-9]|[12]\d|3[01]),\s*\d{4}\s*,\s*"
    r"\d+\s+Stat\.\s*\d+\.?",
    re.I,
)
LABEL = re.compile(r"\(([A-Za-z0-9]+)\)")
ROOT = re.compile(r"(?m)^[ \t]*\(([a-z])\)[ \t]+")
# A numeric code identifier may carry nested labels. Bare attached labels
# require the explicit statutory-reference introducer at this position.
REF = re.compile(
    r"\b(?:sections?|subsections?|paragraphs?|subparagraphs?|clauses?)\s+"
    r"(?:[0-9]+(?:\.[0-9]+)*(?:\([A-Za-z0-9]+\))*|(?:\([A-Za-z0-9]+\))+)"
    r"(?:\s+(?:and|or)\s+(?:[0-9]+(?:\([A-Za-z0-9]+\))*|(?:\([A-Za-z0-9]+\))+))*",
    re.I,
)
MONEY = re.compile(r"\$(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
WORD = re.compile(r"[A-Za-z]+(?:[’'-][A-Za-z]+)*")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
ROMANS = (
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
)


class Refusal(ValueError):
    pass


@dataclass(frozen=True)
class Certificate:
    source_sha256: str
    scope: tuple[str, str]
    parent_span: tuple[int, int]
    notice_span: tuple[int, int]
    next_sibling_start: int
    structural_paths: tuple
    token_spans: tuple


def _reference(text, start, end):
    match = REF.match(text, start, end)
    if match is None:
        return None
    prefix = text[max(0, start - 48) : start]
    suffix = text[match.end() : end]
    if re.search(
        r"\b(?:in|under|by|of|to|from|for|with|and|or|if|as|than)\s+$", prefix, re.I
    ) or re.match(
        r"\s+(?:applies|applied|would|provides|shall|is|means)\b", suffix, re.I
    ):
        return match
    return None


def _prose(text, start, end, tokens, *, parentheses=False):
    """Every delimiter in ordinary parentheticals needs a separate role."""
    i = start
    words = 0
    while i < end:
        if text[i].isspace():
            if text[i] in "\r\n":
                raise Refusal(f"physical discontinuity at {i}")
            i += 1
            continue
        ref = _reference(text, i, end)
        if ref:
            tokens.append((i, ref.end(), "statutory_reference"))
            i = ref.end()
            continue
        money = MONEY.match(text, i, end)
        if money:
            tokens.append((i, money.end(), "currency"))
            i = money.end()
            continue
        if text[i] == "(":
            depth = 1
            j = i + 1
            while j < end and depth:
                depth += (text[j] == "(") - (text[j] == ")")
                j += 1
            if depth:
                raise Refusal(f"unclosed parenthesis at {i}")
            if LABEL.fullmatch(text[i:j]):
                raise Refusal(f"unattributed parenthesized label at {i}")
            _prose(text, i + 1, j - 1, tokens, parentheses=True)
            tokens.append((i, j, "balanced_multiword_parenthetical"))
            i = j
            continue
        word = WORD.match(text, i, end)
        if word:
            if word[0].lower() in (
                "sec",
                "section",
                "subsection",
                "paragraph",
                "subparagraph",
                "clause",
            ) and re.match(r"\s*\d", text[word.end() : end]):
                raise Refusal(f"unproved reference or heading at {i}")
            # Standalone titled/roman headings are not ordinary prose.
            tail = text[word.end() : end].lstrip(" ")
            if (word[0].isupper() or word[0].lower() in ("sec", "section")) and tail[
                :1
            ] in (".", ":", ")", "—"):
                raise Refusal(f"potential prose heading at {i}")
            words += 1
            i = word.end()
            continue
        num = NUMBER.match(text, i, end)
        if num:
            if text[num.end() : end].lstrip(" ")[:1] in (":", ")", "—"):
                raise Refusal(f"potential numeric heading at {i}")
            i = num.end()
            continue
        if text[i] in ',.;:!?“”‘’"%/' or text[i] == chr(39):
            i += 1
            continue
        # A chapeau dash is a delimiter only immediately before a separately
        # resolved structural marker; the caller cuts this gap at that marker.
        if text[i] == "—" and not text[i + 1 : end].strip():
            tokens.append((i, end, "chapeau"))
            i = end
            continue
        raise Refusal(f"unaccounted presentation {text[i]!r} at {i}")
    if parentheses and words < 2:
        raise Refusal(f"parenthetical lacks multiword prose at {start}")


def _unquoted_prefix(text, end):
    """Conservatively account for quotation from source origin, including parent.

    Only intra-word apostrophes are lexical apostrophes. Ambiguous unmatched
    possessive/quotation punctuation refuses; no reset at a physical heading.
    """
    stack = []
    for i, char in enumerate(text[:end]):
        if (
            char in ("'", "’")
            and i > 0
            and i + 1 < len(text)
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        ):
            continue
        if char in ("“", "‘"):
            stack.append("”" if char == "“" else "’")
        elif char in ("”", "’"):
            if not stack or stack[-1] != char:
                raise Refusal(f"unmatched quotation before structural marker at {i}")
            stack.pop()
        elif char in ('"', "'"):
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
    if stack:
        raise Refusal("quoted structural marker or parent")


def _certify(text, scope):
    if (
        len(scope) != 2
        or not re.fullmatch("[a-z]", scope[0])
        or not re.fullmatch("[1-9][0-9]*", scope[1])
    ):
        raise Refusal("only physical alpha parent / numeric child supported")
    roots = list(ROOT.finditer(text))
    parents = [m for m in roots if m[1] == scope[0]]
    if len(parents) != 1:
        raise Refusal("missing or duplicate physical parent")
    parent = parents[0]
    pidx = roots.index(parent)
    _unquoted_prefix(text, parent.start())
    # A noninitial physical parent follows a closed preceding paragraph, not
    # a reference/chapeau introducing a quoted or referenced outline.
    preceding = text[: parent.start()].rstrip()
    if pidx == 0 and preceding:
        raise Refusal("unproved material before first physical parent")
    if preceding and not preceding.endswith("."):
        raise Refusal("physical parent lacks closed preceding prose")
    # Require actual sequential neighboring physical roots, not arbitrary
    # parent-shaped prose. The first root must be (a).
    if (
        not roots
        or roots[0][1] != "a"
        or any(
            ord(b[1]) != ord(a[1]) + 1
            for a, b in zip(roots[: pidx + 1], roots[1 : pidx + 2])
        )
    ):
        raise Refusal("physical root sequence not complete and unique")
    pend = roots[pidx + 1].start() if pidx + 1 < len(roots) else len(text)
    notices = [
        m
        for m in BIBLIOGRAPHIC_NOTICE.finditer(text, parent.end(), pend)
        if m["label"] == scope[1]
    ]
    if len(notices) != 1:
        raise Refusal("missing or duplicate complete notice")
    notice = notices[0]
    close = notice.end()
    if text[close : close + 2] == " ]":
        close += 2
    elif text[close : close + 1] == "]":
        close += 1
    after = close
    while after < pend and text[after] == " ":
        after += 1
    successor = LABEL.match(text, after, pend)
    if not successor or successor[1] != str(int(scope[1]) + 1):
        raise Refusal("missing immediate numeric successor")
    if not re.match(r"\s+[A-Za-z]", text[successor.end() : pend]):
        raise Refusal("successor lacks prose")
    if re.match(
        r"\s+(?:see|refer|under|pursuant|according)\b",
        text[successor.end() : pend],
        re.I,
    ):
        raise Refusal("reference-like successor")
    limit = successor.end()
    i = parent.end()
    gap = i
    tokens = []
    paths = []
    active = []
    categories = ["numeric", "alpha", "roman", "upper_roman"]

    def choices(label):
        out = []
        if label.isdigit():
            out.append((0, int(label)))
        if len(label) == 1 and "A" <= label <= "Z":
            out.append((1, ord(label) - 64))
        if label in ROMANS:
            out.append((2, ROMANS.index(label) + 1))
        if label.lower() in ROMANS and label.isupper():
            out.append((3, ROMANS.index(label.lower()) + 1))
        return out

    def heading(label, start, end):
        nonlocal active
        _unquoted_prefix(text, start)
        options = choices(label)
        continuing = [
            (level, value)
            for level, value in options
            if level < len(active) and value == active[level][1] + 1
        ]
        if len(continuing) > 1:
            raise Refusal("ambiguous sequential category")
        if continuing:
            level, value = continuing[0]
        else:
            descending = [
                (level, value)
                for level, value in options
                if level == len(active) and value == 1
            ]
            if len(descending) != 1:
                raise Refusal(
                    f"gapped/duplicate/reordered/unknown label {label} at {start}"
                )
            level, value = descending[0]
        active = active[:level] + [(label, value)]
        path = (scope[0],) + tuple(a for a, _ in active)
        paths.append((start, end, path))
        tokens.append((start, end, "outline_" + categories[level]))

    while i < limit:
        bib = BIBLIOGRAPHIC_NOTICE.match(text, i, limit)
        if text[i : i + 1] == "[":
            bib = BIBLIOGRAPHIC_NOTICE.match(text, i + 1, limit)
        if bib:
            _prose(text, gap, i, tokens)
            heading(bib["label"], bib.start(), bib.start() + len(bib["label"]) + 2)
            b = bib.end()
            while b < limit and text[b] == " ":
                b += 1
            if b < limit and text[b] == "]":
                b += 1
            tokens.append((i, b, "closed_bibliography"))
            i = b
            gap = i
            continue
        ref = _reference(text, i, limit)
        if ref:
            i = ref.end()
            continue
        if text[i] == "(":
            m = LABEL.match(text, i, limit)
            if m:
                _prose(text, gap, i, tokens)
                heading(m[1], i, m.end())
                i = m.end()
                gap = i
                continue
            depth = 1
            j = i + 1
            while j < limit and depth:
                depth += (text[j] == "(") - (text[j] == ")")
                j += 1
            if depth:
                raise Refusal("unclosed prose parenthesis")
            i = j
            continue
        i += 1
    _prose(text, gap, limit, tokens)
    if tuple(paths[-1][2]) != (scope[0], str(int(scope[1]) + 1)):
        raise Refusal("successor not direct child")
    target = [p for p in paths if p[2] == scope]
    if len(target) != 1 or target[0][0] != notice.start():
        raise Refusal("target ancestry mismatch")
    return Certificate(
        hashlib.sha256(text.encode()).hexdigest(),
        scope,
        (parent.start(), pend),
        (notice.start(), notice.end()),
        after,
        tuple(paths),
        tuple(tokens),
    )


def confirms_bibliographic_editorial_scope(text: str, scope: tuple[str, ...]) -> bool:
    """Fail closed on presentations outside the finite supported ancestry."""
    try:
        _certify(text, scope)
    except Refusal:
        return False
    return True

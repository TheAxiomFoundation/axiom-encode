"""Withdrawn #1847 coverage heuristic, for offline candidate proposals only.

Copied from b139a2506's validator_pipeline.py. Its ownership guesses are not
approval evidence. No axiom_encode module may import this script.
"""

from __future__ import annotations

import re
import unicodedata

from axiom_encode.harness.source_completeness import (
    _dotted_subsection_boundary_matches,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    _HIGH_SIGNAL_SUBPARAGRAPH_PATTERN,
    _TOP_LEVEL_SOURCE_SUBPARAGRAPH_PATTERN,
    _is_horizontal_space,
)

RETIRED_HEURISTIC_COMMIT = "b139a2506e665f867893635265c931e127dfc777"


class _PhysicalLabel(str):
    """Retain a regex origin while preserving the historical label value."""

    def __new__(cls, value: str, physical_start: int):
        result = super().__new__(cls, value)
        result.physical_start = physical_start
        return result


def _normalized_editorial_source_presentation(source_text: str) -> str:
    """Normalize notice typography while preserving physical line breaks."""
    normalized = re.sub(r"<sup>\s*\d+\s*</sup>|[⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", source_text)
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Cf"
    )
    normalized = re.sub(r"</?(?:i|em|b|strong)>", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[\[\]*_]", "", normalized)
    return normalized


def _source_slice_is_editorial_omission(source_text: str) -> bool:
    # Authenticate the whole slice, not a repeal prefix that could swallow a
    # different following label. Formatting and editorial dates do not change
    # the notice's identity; unfamiliar trailing prose remains mandatory.
    normalized = _normalized_editorial_source_presentation(source_text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    if re.fullmatch(
        r"\([A-Za-z0-9]+\)\s*(?:Repealed|Omitted)\.?"
        r"(?:\s*[.—–-]?\s*(?:Effective|Eff\.)\s+"
        r"(?:[A-Za-z]+\s+)?\d{1,4}(?:[\s,/-]+\d{1,4})*\.?)?"
        r"(?:\s+Pub\.\s*L\.\s*\d+-\d+,\s*title\s+[IVXLCDM\d]+,\s*"
        r"section\s+\d+(?:\([A-Za-z0-9]+\))*\.?)?",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    if re.fullmatch(r"\(?[A-Za-z0-9]+\)?\s*(?:\[?\s*)?\.{3,}(?:\s*\]?)?", normalized):
        return True
    return False


def _high_signal_top_level_subparagraphs(
    source_text: str, *, corpus_citation_path: str = ""
) -> list[tuple[str, str]]:
    return [
        (label, text)
        for label, text in _coverage_source_subparagraph_segments(
            source_text, corpus_citation_path=corpus_citation_path
        )
        if _HIGH_SIGNAL_SUBPARAGRAPH_PATTERN.search(text)
    ]


def _coverage_source_subparagraph_segments(
    source_text: str, *, corpus_citation_path: str = ""
) -> list[tuple[str, str]]:
    matches = list(_TOP_LEVEL_SOURCE_SUBPARAGRAPH_PATTERN.finditer(source_text))
    if not matches:
        return []
    branches = recognize_source_structure(
        source_text, corpus_citation_path=corpus_citation_path
    )
    by_start = {}
    for branch in branches:
        # Attached chains share an offset, e.g. (a)(1)(A). The leading
        # marker owns the coverage label, not its deeper attached children.
        by_start.setdefault(branch.start, branch)
    # A skipped possible marker breaks the proof of ancestry. Deliberately
    # accept more than editorial notices here: uncertainty must retain demands
    # without requiring an exhaustive catalog of extraction typography.
    recognized_marker_starts = set(by_start)
    for item in branches:
        physical = re.match(
            r"[^\S\r\n]*((?:\([A-Za-z0-9]+\)[ \t]*)+)",
            source_text[item.start :],
        )
        if physical:
            recognized_marker_starts.update(
                item.start + physical.start(1) + marker.start()
                for marker in re.finditer(r"\([A-Za-z0-9]+\)", physical[1])
            )
    skipped_markers: list[tuple[int, str]] = []
    for candidate in re.finditer(r"\(([^()\r\n]+)\)", source_text):
        marker_start = candidate.start()
        if marker_start not in recognized_marker_starts:
            skipped_markers.append((marker_start, candidate[1].strip()))
    scope = _source_subparagraph_unit_scope(corpus_citation_path)
    if scope is not None and any(not token.isascii() for token in scope):
        scope = None
    numeric_candidates = [
        branch
        for branch in branches
        if len(branch.path) == 1
        and branch.path[0][0].isdigit()
        and branch.label.startswith("(")
        and source_text[branch.start : branch.start + 1] == "("
    ]
    numeric_outline = bool(numeric_candidates and numeric_candidates[0] == branches[0])
    numeric_roots = numeric_candidates[:1]
    own_numeric_list = False
    previous_letter = -1
    for item in numeric_candidates[1:]:
        preceding_letter = next(
            (
                match.start()
                for match in reversed(matches)
                if match.start() < item.start
            ),
            -1,
        )
        if preceding_letter != previous_letter:
            own_numeric_list = False
            previous_letter = preceding_letter
        if (
            scope
            and item.path == scope[-len(item.path) :]
            and numeric_roots[-1].path != item.path
        ):
            # A requested physical parent cannot be dismissed as an earlier
            # letter's incidental numbered list merely because its number
            # resets. Its direct children must remain coverage obligations.
            numeric_roots.append(item)
            own_numeric_list = False
            continue
        parent_order = re.fullmatch(r"(\d+)([a-z]?)", numeric_roots[-1].path[0])
        item_order = re.fullmatch(r"(\d+)([a-z]?)", item.path[0])
        if own_numeric_list or (
            preceding_letter >= 0
            and parent_order
            and item_order
            and (int(item_order[1]), item_order[2])
            <= (int(parent_order[1]), parent_order[2])
        ):
            # A letter's own numbered list resets to (1). Its later (2), etc.
            # remain children even when extraction removes their indentation.
            own_numeric_list = True
            continue
        numeric_roots.append(item)
    dotted_boundaries = _dotted_subsection_boundary_matches(source_text)
    possible_heading_starts = _coverage_unproven_prose_starts(
        source_text,
        recognized_marker_starts=recognized_marker_starts,
        numeric_outline=numeric_outline,
    )
    children: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = _PhysicalLabel(match.group("label").lower(), match.start())
        branch = by_start.get(match.start())
        ancestry = None
        uncertain_ancestry = False
        if branch is not None:
            # Repeated attached ancestors may be omitted by the recognizer.
            # Align its first emitted branch with the physical marker chain,
            # so (a)(1) still belongs to the leading (a), even if only (1)
            # was emitted here. Ambiguous alignment cannot prove nesting.
            marker_chain = re.match(
                r"(?:\([A-Za-z0-9]+\)[ \t]*)+", source_text[match.start() :]
            )
            marker_labels = re.findall(r"\(([A-Za-z0-9]+)\)", marker_chain[0])
            alignments = [
                count
                for count in range(1, len(marker_labels) + 1)
                if branch.label == f"({marker_labels[count - 1]})"
                and branch.path[-count:]
                == tuple(item.lower() for item in marker_labels[:count])
            ]
            ancestry = branch.path[: -alignments[0]] if len(alignments) == 1 else None
            if numeric_outline and ancestry == ():
                # The recognizer deliberately falls back to flat paragraphs
                # for a partial numeric outline, e.g. an isolated (5)(c).
                # Its recognized numeric heading still owns the lettered list.
                parent = next(
                    (
                        item
                        for item in reversed(numeric_roots)
                        if item.start < match.start()
                    ),
                    None,
                )
                if parent is not None:
                    ancestry = parent.path
            scoped_child = (
                ancestry
                and scope is not None
                and len(ancestry) <= len(scope)
                and ancestry == scope[-len(ancestry) :]
            )
            ancestor_start = min(
                (
                    item.start
                    for item in branches
                    if ancestry
                    and item.start < match.start()
                    and len(item.path) <= len(ancestry)
                    and ancestry[: len(item.path)] == item.path
                ),
                default=match.start(),
            )
            uncertain_ancestry = (
                any(
                    ancestor_start < position < match.start()
                    # Only a positively authenticated single-letter omission can
                    # preserve numeric ancestry. An arbitrary alphabetic marker
                    # can introduce an unknown parent just as a numeric one can.
                    and (
                        not numeric_outline
                        or not re.fullmatch(r"[a-z]", skipped_label)
                        or not _source_slice_is_editorial_omission(
                            source_text[position : match.start()]
                        )
                    )
                    for position, skipped_label in skipped_markers
                )
                or any(
                    ancestor_start < position < match.start()
                    for position in possible_heading_starts
                )
            )
            physical_parent = any(
                ancestry
                and item.path == ancestry[:1]
                and item.start < match.start()
                and re.match(
                    r"[^\S\r\n]*(?:\([0-9]+[a-z]?\)|[A-Z]\.)",
                    source_text[item.start :],
                )
                for item in branches
            )
            # A flattened letter-root outline can leave a roman ancestor
            # active even without an editorial notice. Its path alone cannot
            # distinguish a new root letter from a genuine nested child.
            if (
                ancestry
                and all(token.isascii() for token in ancestry)
                and physical_parent
                and scope is not None
                and not scoped_child
                and not uncertain_ancestry
            ):
                continue
        # Unrecognized markers remain demands: incomplete recognition is not
        # evidence that real source content may be dropped. Keep the original
        # signal-bearing span, including a root's indented numbered children.
        start = match.start("text")
        end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(source_text)
        )
        original_end = end
        end = next(
            (item.start() for item in dotted_boundaries if start < item.start() < end),
            end,
        )
        if numeric_outline:
            end = next(
                (item.start for item in numeric_roots if start < item.start < end),
                end,
            )
        if (
            end < original_end
            and not _HIGH_SIGNAL_SUBPARAGRAPH_PATTERN.search(source_text[start:end])
            and (
                scope is None
                or not _coverage_borrowed_signal_is_nested(
                    source_text[end:original_end]
                )
            )
        ):
            # An isolated dotted heading cannot prove the depth of all later
            # content. Retain a root's legacy demand if that ownership is unclear.
            end = original_end
        text = source_text[start:end].strip()
        children.append((label, text))
    return children


def _coverage_unproven_prose_starts(
    source_text: str,
    *,
    recognized_marker_starts: set[int],
    numeric_outline: bool = False,
) -> set[int]:
    """Require positive prose evidence between authenticated physical markers.

    A recognizer's silence cannot prove continuity. Classify each presentation
    token as ordinary prose or an already authenticated marker; everything else
    interrupts ancestry. Positions remain physical so uncertainty after a cue
    cannot invalidate that cue's earlier owner.
    """
    starts: set[int] = set()
    prose_punctuation = frozenset(".,;:!?\"'“”‘’%/\\—–-")
    label_delimiters = frozenset(".):—–-")
    ordinary_clause = re.compile(
        r"[^\W\d_]+(?:\s+[^\W\d_]+){0,12}[\"'”’]?\s+"
        r"(?:means|is|are|was|were|has|have|must|shall|will|can|may|does|do|did|"
        r"continues|provides|follows|applies|contains|includes|requires|specifies)\s+\S",
        flags=re.IGNORECASE,
    )
    for line in re.finditer(r"(?m)^.*$", source_text):
        characters: list[str] = []
        offsets: list[int] = []
        emphasis_positions: set[int] = set()
        for emphasis in re.finditer(r"(\*{1,2}|_|`)(?=\S).*?\S\1", line[0]):
            emphasis_positions.update(range(emphasis.start(), emphasis.end(1)))
            emphasis_positions.update(
                range(emphasis.end() - len(emphasis[1]), emphasis.end())
            )
        for token in re.finditer(
            r"</?(?:i|em|b|strong|sup)>|.", line[0], flags=re.IGNORECASE
        ):
            if token[0].startswith("<") and token[0].endswith(">"):
                continue
            if token.start() in emphasis_positions:
                continue
            if token[0].isspace() and not _is_horizontal_space(token[0]):
                if token[0] != "\r" or token.end() != len(line[0]):
                    starts.add(line.start() + token.start())
            # Compatibility normalization can turn an unfamiliar symbol into
            # ordinary letters. Its original presentation still lacks prose
            # proof (e.g. a square unit symbol flattened into a word).
            if unicodedata.category(token[0]).startswith("S"):
                starts.add(line.start() + token.start())
            for char in unicodedata.normalize("NFKC", token[0]):
                if unicodedata.category(char) == "Cf":
                    continue
                characters.append(char)
                offsets.append(line.start() + token.start())
        presentation = "".join(characters)
        authenticated: set[int] = set()
        raw_physical = re.match(r"[^\S\r\n]*((?:\([A-Za-z0-9]+\)[ \t]*)+)", line[0])
        if raw_physical and (
            line.start() in recognized_marker_starts
            or line.start() + raw_physical.start(1) in recognized_marker_starts
        ):
            authenticated.update(
                range(
                    line.start() + raw_physical.start(1),
                    line.start() + raw_physical.end(1),
                )
            )
        physical = re.match(r"[^\S\r\n]*((?:\([A-Za-z0-9]+\)[ \t]*)+)", presentation)
        if physical:
            # Stripping a wrapper must not attach an unknown inline marker to
            # a recognized prefix. Formatted markers need their own identity.
            for marker in re.finditer(r"\([A-Za-z0-9]+\)", physical[1]):
                start = physical.start(1) + marker.start()
                end = physical.start(1) + marker.end()
                if offsets[start] in recognized_marker_starts:
                    authenticated.update(offsets[start:end])
        if numeric_outline:
            # This exception is positive whole-notice proof, not a guessed
            # heading shape. It preserves established numbered-outline behavior.
            for opening in re.finditer(r"\(", presentation):
                remainder = presentation[opening.start() :]
                if re.match(
                    r"\([a-z]\)", remainder
                ) and _source_slice_is_editorial_omission(remainder):
                    authenticated.update(
                        offsets[index]
                        for index in range(opening.start(), len(offsets))
                        if presentation[index] not in "[]"
                    )
                    bracket_stack: list[int] = []
                    for index, char in enumerate(presentation):
                        if char == "[":
                            bracket_stack.append(index)
                        elif char == "]" and bracket_stack:
                            bracket_start = bracket_stack.pop()
                            content = presentation[bracket_start + 1 : index].strip()
                            notice_wrapper = _source_slice_is_editorial_omission(
                                content
                            )
                            date_wrapper = re.match(
                                r"(?:Effective|Eff\.)\s", content, flags=re.IGNORECASE
                            ) and _source_slice_is_editorial_omission(
                                "(b) Repealed. " + content
                            )
                            if notice_wrapper or date_wrapper:
                                authenticated.update(
                                    (offsets[bracket_start], offsets[index])
                                )
                    break
        leading = len(presentation) - len(presentation.lstrip())
        label_leading = re.match(r"[\s\"'“”‘’]*", presentation).end()
        if leading < len(presentation) and offsets[leading] not in authenticated:
            # A punctuated title is still uncertain. A plain continuation
            # needs an ordinary subject/predicate clause and its sentence end.
            clause = ordinary_clause.match(presentation, label_leading)
            sentence_end = (
                re.search(r"[.!?](?=[\s\"'”’]|$)", presentation[clause.end() :])
                if clause
                else None
            )
            if not sentence_end:
                starts.add(offsets[leading])
            else:
                end = clause.end() + sentence_end.end()
                remainder = presentation[end:].strip(" \t\"'“”‘’")
                if remainder:
                    # A later fragment cannot invalidate an earlier cue.
                    starts.add(offsets[end])
        for token in re.finditer(r"\w+|[^\w\s]", presentation):
            position = offsets[token.start()]
            if position in authenticated:
                continue
            word = token[0]
            if word.isalnum():
                following = presentation[token.end() :].lstrip()
                original_word = line[0][
                    offsets[token.start()] - line.start() : offsets[token.end() - 1]
                    - line.start()
                    + 1
                ]
                section_word = word.lower().startswith("sec")
                line_leading = token.start() == label_leading
                label_word = (
                    len(word) == 1
                    or len(original_word) == 1
                    or any(char.isdigit() for char in word)
                    or word.isupper()
                    or all(char.upper() in "IVXLCDM" for char in word)
                )
                if (
                    section_word
                    or (following and following[0] == ")")
                    or (
                        (label_word or line_leading)
                        and (
                            (label_word and line_leading)
                            or (following and following[0] in label_delimiters)
                        )
                    )
                ):
                    starts.add(position)
                # Other alphabetic words and undelimited inline numbers are
                # ordinary prose. Nonalphabetic/nondecimal word tokens have no
                # positive prose interpretation (e.g. unusual numeric symbols).
                elif not (word.isalpha() or word.isdecimal()):
                    starts.add(position)
            elif word in prose_punctuation:
                if word in "—–-" and not (
                    token.start() > leading
                    and presentation[token.start() - 1].isalpha()
                    and token.end() < len(presentation)
                    and presentation[token.end()].isalpha()
                ):
                    starts.add(position)
            else:
                # Opening delimiters, section signs, bullets, and unfamiliar
                # symbols are not affirmative ordinary-prose evidence.
                starts.add(position)
    return starts


def _coverage_borrowed_signal_is_nested(source_text: str) -> bool:
    """Prove each borrowed cue is within a physical numbered heading's body."""
    headings = list(
        re.finditer(
            r"(?m)^[ \t]*(?:\((?P<number>[0-9]+[a-z]?)\)|"
            r"\([A-Za-z0-9]+\)|(?:[A-Za-z]|[A-Z]+|[ivxlcdm]+)\.)",
            source_text,
        )
    )
    possible_boundaries = _coverage_unproven_prose_starts(
        source_text, recognized_marker_starts={heading.start() for heading in headings}
    )
    for signal in _HIGH_SIGNAL_SUBPARAGRAPH_PATTERN.finditer(source_text):
        heading = next(
            (item for item in reversed(headings) if item.start() < signal.start()),
            None,
        )
        # Inline parentheses in a dotted heading may be a citation, and a
        # subsequent letter/roman heading may start a separate outline.
        if (
            heading is None
            or not heading.group("number")
            or any(
                heading.start() < boundary <= signal.start()
                for boundary in possible_boundaries
            )
        ):
            return False
    return True


def _source_subparagraph_unit_scope(citation_path: str) -> tuple[str, ...] | None:
    """Identify subsection tokens, or return None when the root is uncertain."""
    if not citation_path.strip("/"):
        return ()
    parts = citation_path.strip("/").split("/")
    if len(parts) < 3:
        return None
    if parts[1] == "statute" and (parts[0] == "us" or parts[0].startswith("us-")):
        section_end = 4
        if parts[0].startswith("us-") and re.search(r"[.:-]", parts[2]):
            # Flat state section identifiers: 206.30, 47:302, 7-2-18.13.
            section_end = 3
    elif parts[:2] == ["us", "regulation"] and parts[2].isdigit():
        if len(parts) >= 4 and re.fullmatch(r"\d+\.[A-Za-z0-9.-]+", parts[3]):
            # Combined CFR part/section, e.g. 26/1.36B-2.
            section_end = 4
        elif len(parts) >= 5 and parts[3].isdigit():
            section_end = 5
        else:
            return None
    elif (
        parts[:2] == ["us-co", "regulation"]
        and len(parts) >= 4
        and re.fullmatch(r"\d+-ccr-\d+-\d+", parts[2])
        and re.fullmatch(r"\d+(?:\.\d+)+", parts[3])
    ):
        section_end = 4
    elif parts[1] == "regulation" and (
        (parts[0] == "us-ny" and re.fullmatch(r"\d+-nycrr", parts[2]))
        or (
            parts[0] == "us-wa"
            and len(parts) >= 5
            and parts[3].startswith(f"{parts[2]}-")
            and parts[4].startswith(f"{parts[3]}-")
        )
    ):
        # NYCRR title/part/section and WAC title/chapter/section roots.
        section_end = 5
    else:
        return None
    if len(parts) < section_end or not all(parts):
        return None
    return tuple(part.lower() for part in parts[section_end:])

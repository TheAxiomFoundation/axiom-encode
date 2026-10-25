"""Text-level reading of recorded validator issues.

Recorded issues are plain strings today (``validation-issue/v1`` from #1660
is not merged). This module reads only what the text states: an optional
``<file>.yaml: <gate>:`` prefix, a leading bracket tag, and, for the
persistence probe, a signature with volatile parts (line numbers, offsets,
digests, absolute paths) replaced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FILE_PREFIX = re.compile(
    r"^(?P<file>[^\s:`]+\.ya?ml): (?:(?P<gate>[a-z][a-z0-9_-]*): )?"
)
_LEADING_TAG = re.compile(r"^\[(?P<tag>[a-z][a-z0-9_.-]*(?::[a-z0-9_.-]+)*)\]")

_VOLATILE: tuple[tuple[re.Pattern[str], str], ...] = (
    # Absolute and placeholder roots in front of a RuleSpec checkout path.
    (
        re.compile(r"(?:<rulespec-validation-root>|/[^\s`'\"]*?)/(rulespec-[a-z]+)/"),
        r"\1/",
    ),
    (re.compile(r"(?<![\w.])/(?:tmp|private|home|Users|var)/[^\s`'\"]+"), "<path>"),
    (re.compile(r"\b(line|column|col|row)\s+\d+", re.IGNORECASE), r"\1 #"),
    (re.compile(r"\boffsets?\s+\d+:\d+", re.IGNORECASE), "offsets #:#"),
    (re.compile(r"\bat\s+\d+:\d+\b"), "at #:#"),
    # Digests only: a hex run with a letter, not part of a decimal number.
    (re.compile(r"(?<![\d.])\b(?=[0-9]*[a-f])[0-9a-f]{12,64}\b"), "<hex>"),
    (re.compile(r"\s+"), " "),
)
MAX_SIGNATURE_CHARS = 300


@dataclass(frozen=True)
class ParsedIssue:
    text: str
    file: str | None
    gate: str | None
    tag: str | None
    body: str


def parse_issue(text: str) -> ParsedIssue:
    """Split one recorded issue into its file prefix, gate, tag and body."""

    value = str(text)
    file = gate = None
    body = value.lstrip()
    match = _FILE_PREFIX.match(body)
    if match is not None:
        file = match.group("file")
        gate = match.group("gate")
        body = body[match.end() :]
    tag_match = _LEADING_TAG.match(body)
    return ParsedIssue(
        text=value,
        file=file,
        gate=gate,
        tag=tag_match.group("tag") if tag_match else None,
        body=body,
    )


def issue_signature(text: str, max_chars: int | None = MAX_SIGNATURE_CHARS) -> str:
    """Return the issue text with volatile parts normalized.

    Comparisons use the whole signature (``max_chars=None``); the bounded
    form is only for display.
    """

    signature = str(text).strip()
    for pattern, replacement in _VOLATILE:
        signature = pattern.sub(replacement, signature)
    signature = signature.strip()
    return signature if max_chars is None else signature[:max_chars]

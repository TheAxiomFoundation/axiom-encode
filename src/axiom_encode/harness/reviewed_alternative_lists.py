"""Reviewed exceptions for bulleted lists of alternatives, bound to exact text.

There is deliberately no source interpretation here. The gates-collapsed check
reads every bullet under a condition cue as one more simultaneous fact. For a
list of alternatives that reading is wrong. A registry entry records, for one
source unit and one exact text, the clause that a reviewer read as a list of
alternatives. For that clause the check is switched off: it verifies nothing
about the clause, and the entry says in plain words what it no longer verifies
and what the approving owner verified instead.

An entry applies only when the citation, the SHA-256 of the text and every
recorded line at its recorded offset match. Offsets count Python Unicode
characters; hashes cover UTF-8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping

REGISTRY_PATH = "data/reviewed_alternative_lists.jsonl"

# How the reviewer read one physical line after the chapeau.
#   alternative        one alternative of the list
#   outside_condition  text after the list that is not part of the condition
READINGS = frozenset({"alternative", "outside_condition"})

# ``reviewer`` is the owner who checked every recorded line against the source
# text: a GitHub login. Until an owner has done that, it is this marker and
# ``reviewed_at`` is null. The approving owner replaces both in the pull
# request that adds the entry, before merge.
PENDING_REVIEWER = "pending-owner-review"
_GITHUB_LOGIN = re.compile(r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")


def scanned_source_text(source_text: str) -> str:
    """Return the text that the check scans and that the digest covers.

    ``analyze_complete_source_unit`` strips the unit text before any check
    reads it. Offsets count characters of this stripped text.
    """

    return source_text.strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _literal(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "*?[]"):
        raise ValueError(f"{field} must be a nonempty literal without wildcards")
    return value


def _digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase sha256")
    return value


def _one_line(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{field} must be one nonblank physical line")
    return value


@dataclass(frozen=True)
class ReviewedLine:
    """One physical source line at an exact character offset."""

    physical_start: int
    line: str
    reading: str

    @property
    def physical_end(self) -> int:
        return self.physical_start + len(self.line)

    @classmethod
    def from_mapping(cls, value: Any, field: str, *, chapeau: bool) -> ReviewedLine:
        fields = {"physical_start", "line"} | (set() if chapeau else {"reading"})
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"{field} has missing or unknown fields")
        start = value["physical_start"]
        if type(start) is not int or start < 0:
            raise ValueError(f"{field}.physical_start must be an integer >= 0")
        reading = "chapeau" if chapeau else value["reading"]
        if not chapeau and reading not in READINGS:
            raise ValueError(f"{field}.reading is unknown")
        return cls(start, _one_line(value["line"], f"{field}.line"), reading)

    def is_in(self, source_text: str) -> bool:
        """The line stands at its offset and is a whole physical line."""

        return (
            source_text[self.physical_start : self.physical_end] == self.line
            and (
                self.physical_start == 0 or source_text[self.physical_start - 1] == "\n"
            )
            and (
                self.physical_end == len(source_text)
                or source_text[self.physical_end] == "\n"
            )
        )


@dataclass(frozen=True)
class ReviewedAlternativeList:
    citation_path: str
    source_text_sha256: str
    chapeau: ReviewedLine
    lines: tuple[ReviewedLine, ...]
    reason: str
    not_examined: tuple[str, ...]
    owner_must_verify: tuple[str, ...]
    author: str
    reviewer: str
    reviewed_at: str | None

    @classmethod
    def from_mapping(cls, value: Any) -> ReviewedAlternativeList:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("reviewed alternative list has missing or unknown fields")
        _literal(value["citation_path"], "citation_path")
        _digest(value["source_text_sha256"], "source_text_sha256")
        chapeau = ReviewedLine.from_mapping(value["chapeau"], "chapeau", chapeau=True)
        raw_lines = value["lines"]
        if not isinstance(raw_lines, (list, tuple)):
            raise ValueError("lines must be a list")
        lines = tuple(
            ReviewedLine.from_mapping(line, f"lines[{index}]", chapeau=False)
            for index, line in enumerate(raw_lines)
        )
        in_list = [line for line in lines if line.reading == "alternative"]
        if len(in_list) < 2:
            raise ValueError("a list of alternatives needs at least two alternatives")
        if any(line.reading != "alternative" for line in lines[: len(in_list)]):
            raise ValueError(
                "outside_condition lines may only follow the last line of the list"
            )
        previous = chapeau
        for line in lines:
            if line.physical_start <= previous.physical_end:
                raise ValueError("lines must follow the chapeau in source order")
            previous = line
        statements = {}
        for field in ("not_examined", "owner_must_verify"):
            raw = value[field]
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError(f"{field} must be a nonempty list")
            statements[field] = tuple(
                _one_line(item, f"{field}[{index}]") for index, item in enumerate(raw)
            )
        for field in ("reason", "author"):
            _one_line(value[field], field)
        reviewer, reviewed_at = value["reviewer"], value["reviewed_at"]
        if reviewer == PENDING_REVIEWER:
            if reviewed_at is not None:
                raise ValueError("a pending review has no reviewed_at")
        elif not isinstance(reviewer, str) or not _GITHUB_LOGIN.fullmatch(reviewer):
            raise ValueError(
                f"reviewer must be a GitHub login (@name) or {PENDING_REVIEWER!r}"
            )
        elif not isinstance(reviewed_at, str):
            raise ValueError("reviewed_at is required with a reviewer")
        else:
            date.fromisoformat(reviewed_at)
        return cls(
            value["citation_path"],
            value["source_text_sha256"],
            chapeau,
            lines,
            value["reason"],
            statements["not_examined"],
            statements["owner_must_verify"],
            value["author"],
            value["reviewer"],
            value["reviewed_at"],
        )

    @property
    def key(self) -> tuple[str, str, int]:
        return (
            self.citation_path,
            self.source_text_sha256,
            self.chapeau.physical_start,
        )

    def matches(self, citation_path: str, source_text: str) -> bool:
        """Citation, text digest and every recorded line agree with the source."""

        return (
            citation_path == self.citation_path
            and text_sha256(source_text) == self.source_text_sha256
            and self.chapeau.is_in(source_text)
            and all(line.is_in(source_text) for line in self.lines)
        )


def parse_reviewed_alternative_lists(raw: str) -> tuple[ReviewedAlternativeList, ...]:
    """Parse registry text. Any malformed or duplicate entry is an error."""

    entries = tuple(
        ReviewedAlternativeList.from_mapping(json.loads(line))
        for line in raw.splitlines()
        if line
    )
    if len({entry.key for entry in entries}) != len(entries):
        raise ValueError("duplicate reviewed alternative list key")
    return entries


@lru_cache(maxsize=1)
def load_reviewed_alternative_lists() -> tuple[ReviewedAlternativeList, ...]:
    """Load the packaged registry, strictly. Tests and CI call this."""

    return parse_reviewed_alternative_lists(
        files("axiom_encode.harness")
        .joinpath(REGISTRY_PATH)
        .read_text(encoding="utf-8")
    )


def reviewed_alternative_lists_for(
    citation_path: str | None,
    source_text: str,
    *,
    entries: tuple[ReviewedAlternativeList, ...] | None = None,
) -> tuple[ReviewedAlternativeList, ...]:
    """Return the entries that are bound to this citation and this exact text.

    This fails closed: without a citation, with an unreadable or malformed
    registry, or on any mismatch, there is no entry and the check keeps every
    gate. ``tests/test_reviewed_alternative_lists.py`` makes a malformed
    committed registry fail loudly.
    """

    if not citation_path:
        return ()
    if entries is None:
        try:
            entries = load_reviewed_alternative_lists()
        except Exception:
            # Fail closed on any load error, a RecursionError from a hostile
            # line included.
            return ()
    return tuple(
        entry for entry in entries if entry.matches(citation_path, source_text)
    )


def owner_review_is_required(environ: Mapping[str, str] | None = None) -> bool:
    """Say whether a pending owner review must fail in this run.

    It follows the draft gate of ``.github/workflows/ci.yml``: a draft pull
    request and a local run may hold pending entries. A pull request that is
    ready for review, the merge queue and a push to ``main`` may not. A pull
    request whose event payload cannot be read counts as ready.
    """

    environ = os.environ if environ is None else environ
    event = environ.get("GITHUB_EVENT_NAME", "")
    if event == "merge_group":
        return True
    if event == "push":
        return environ.get("GITHUB_REF", "") == "refs/heads/main"
    if event not in {"pull_request", "pull_request_target"}:
        return False
    try:
        with open(environ["GITHUB_EVENT_PATH"], encoding="utf-8") as handle:
            draft = json.load(handle)["pull_request"]["draft"]
    except Exception:
        return True
    return draft is not True


def main(argv: list[str] | None = None) -> int:
    """Print the digest and the line offsets that an entry must record."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("source", help="file that holds the unit text")
    args = parser.parse_args(argv)
    with open(args.source, encoding="utf-8") as handle:
        text = scanned_source_text(handle.read())
    print(f"source_text_sha256 {text_sha256(text)}")
    offset = 0
    for line in text.split("\n"):
        if line.strip():
            print(f"{offset}\t{json.dumps(line, ensure_ascii=False)}")
        offset += len(line) + 1
    return 0

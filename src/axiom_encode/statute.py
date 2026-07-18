"""Helpers for parsing USC citations and deriving RuleSpec paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_RULESPEC_PATH_DASH_TRANSLATION = str.maketrans(
    {
        ord(character): "-"
        for character in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
    }
)


def normalize_rulespec_path_segment(segment: str) -> str:
    """Return an ASCII-dash filesystem segment without changing legal IDs."""

    return segment.translate(_RULESPEC_PATH_DASH_TRANSLATION)


@dataclass(frozen=True)
class CitationParts:
    """Structured representation of a USC citation."""

    title: str
    section: str
    fragments: tuple[str, ...] = ()


def parse_usc_citation(citation: str) -> CitationParts:
    """Parse a USC citation or slash path into title, section, and fragments."""
    cleaned = citation.strip().replace("§", "")
    for prefix in ("us:statutes/", "us:statute/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned.removeprefix(prefix)
            break

    if "/" in cleaned and "USC" not in cleaned.upper():
        parts = [part for part in cleaned.split("/") if part]
        if parts[:2] in (["us", "statute"], ["us", "statutes"]):
            parts = parts[2:]
        elif parts and parts[0] == "statutes":
            parts = parts[1:]
        if len(parts) < 2:
            raise ValueError(f"Could not parse citation: {citation}")
        return CitationParts(
            title=parts[0],
            section=parts[1],
            fragments=tuple(parts[2:]),
        )

    cleaned = re.sub(r"\bUSC\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    pieces = cleaned.split(" ", 1)
    if len(pieces) != 2:
        raise ValueError(f"Could not parse citation: {citation}")

    title = pieces[0]
    remainder = pieces[1].replace(" ", "")
    match = re.match(
        r"^(?P<section>[0-9A-Za-z.-]+)(?P<tail>(?:\([^)]+\))*)$", remainder
    )
    if not match:
        raise ValueError(f"Could not parse citation: {citation}")

    fragments = tuple(re.findall(r"\(([^)]+)\)", match.group("tail")))
    return CitationParts(
        title=title, section=match.group("section"), fragments=fragments
    )


def citation_to_citation_path(citation: str | CitationParts) -> str:
    """Convert a citation into Axiom's canonical citation path format."""
    parts = (
        citation
        if isinstance(citation, CitationParts)
        else parse_usc_citation(citation)
    )
    path_parts = [parts.title, parts.section, *parts.fragments]
    return "us/statute/" + "/".join(path_parts)


def citation_to_relative_rulespec_path(citation: str | CitationParts) -> Path:
    """Convert a citation into the repo-relative RuleSpec output path."""
    parts = (
        citation
        if isinstance(citation, CitationParts)
        else parse_usc_citation(citation)
    )
    title = normalize_rulespec_path_segment(parts.title)
    section = normalize_rulespec_path_segment(parts.section)
    fragments = tuple(normalize_rulespec_path_segment(item) for item in parts.fragments)
    section_path = Path("statutes") / title / section
    if not fragments:
        return section_path.with_suffix(".yaml")
    return section_path / Path(*fragments).with_suffix(".yaml")

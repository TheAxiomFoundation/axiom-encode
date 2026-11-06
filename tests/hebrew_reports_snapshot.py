"""Fresh counts for labelled reports refusals; the external statute snapshot is optional."""

from __future__ import annotations

import bisect
import functools
import json
import os
import re
from pathlib import Path

import pytest

from axiom_encode.harness import hebrew_source_text as hebrew


@functools.lru_cache(maxsize=1)
def snapshot_reports_occurrences() -> tuple[dict, ...]:
    path = Path(
        os.environ.get(
            "AXIOM_HEBREW_STATUTE_SNAPSHOT",
            str(
                Path.home()
                / "TheAxiomFoundation/ops-lanes/il-lane/contributions/corpus-snapshot/2026-09-29-il-taxben-contributions.jsonl"
            ),
        )
    )
    if not path.is_file():
        pytest.skip(f"statute snapshot unavailable: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1633
    rows = []
    for record in records:
        for field in ("body", "heading"):
            text = hebrew.fold_marks(record.get(field) or "")
            stops = hebrew._clause_stops(text)
            links = []
            if field == "body":
                citation = record.get("citation_path") or ""
                for cited, _ in hebrew._section_citations(
                    text,
                    hebrew.current_instrument_kind(citation),
                    hebrew.SOURCE_SECTION_DEPENDENCY,
                ):
                    start = hebrew._clause_start(stops, cited.start())
                    linker = hebrew._source_linker(
                        text,
                        max(start, cited.start() - hebrew._LINK_WINDOW),
                        cited.end("proclitics"),
                    )
                    if linker is not None:
                        links.append((start, linker.start()))
            for match in re.finditer(r"[א-ת]*דוחות[א-ת]*", text):
                stop = bisect.bisect_left(stops, match.end())
                end = stops[stop] if stop < len(stops) else len(text)
                start = hebrew._clause_start(stops, match.start())
                rows.append(
                    {
                        "citation": record.get("citation_path"),
                        "word": match.group(),
                        "clause": text[start:end],
                        "text": text,
                        "start": match.start(),
                        "end": match.end(),
                        "guarded": any(
                            a <= match.start() < match.end() <= b for a, b in links
                        ),
                    }
                )
    assert len(rows) == 25
    return tuple(rows)


def guarded_snapshot_count(source: str) -> int:
    """Count the row's reports construction before a production guarded link.

    A literal clause prefix, from its first reports token to the supplied
    accordance linker, identifies the construction rather than its branch
    label or an invented count. Whitespace is folded as in the source reader.
    """
    source = hebrew.fold_marks(source)
    match = re.search(r"[א-ת]*דוחות[א-ת]*", source)
    assert match is not None, source
    end = source.index("בהתאם לסעיף", match.end())
    # Include the immediately preceding word to distinguish filing contexts.
    previous = re.search(r"[א-ת]+\s+$", source[: match.start()])
    start = previous.start() if previous else match.start()
    construction = " ".join(source[start:end].split())
    return sum(
        row["guarded"] and construction in " ".join(row["clause"].split())
        for row in snapshot_reports_occurrences()
    )
